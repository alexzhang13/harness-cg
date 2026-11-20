"""LLM judge for generated rollouts (rubric "reverify").

Per rollout: a quality verdict, a DERIVED / LUCKY check (was the answer found from visible evidence or guessed), and a
right-reason check when the env exposes option values. Unparseable verdicts are rejected.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .derived_rubric import RUBRIC_SYSTEM as DERIVED_SYSTEM, build_user_prompt as build_derived_prompt
from .filters import code_blocks
from .judge_variant import PROBE_FROM_PROMPT_FLAG, rubric_text

RUBRIC_DIR = Path(__file__).resolve().parent / "rubric"
QUALITY_SYSTEM = "\n".join(rubric_text(p).rstrip("\n") for p in sorted(RUBRIC_DIR.glob("*.md")) if p.name != "README.md") + "\n"


@dataclass
class JudgeCfg:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    headers: dict[str, str] | None = None
    max_tokens: int = 4000
    timeout_s: float = 120.0

    @classmethod
    def from_env(cls, model: str | None = None, base_url: str | None = None) -> "JudgeCfg":
        """Prime Inference by default (PRIME_API_KEY, plus PRIME_API_TEAM if set)."""
        base_url = base_url or os.environ.get("RLM_JUDGE_BASE_URL") or "https://api.pinference.ai/api/v1"
        model = model or os.environ.get("RLM_JUDGE_MODEL") or "openai/gpt-5.6-sol"
        key = os.environ.get("RLM_JUDGE_API_KEY") or os.environ.get("PRIME_API_KEY") or "EMPTY"
        headers = {}
        team = os.environ.get("PRIME_API_TEAM")
        if team and "pinference" in base_url:
            headers["X-Prime-Team-ID"] = team
        return cls(base_url=base_url, model=model, api_key=key, headers=headers or None)


def render_subcalls(row: dict, *, max_entries: int = 6, chars: int = 900) -> str:
    log = row.get("sub_calls") or []
    if not log:
        return "(no sub-calls)"
    parts = [f"{len(log)} sub-call(s) recorded; prompt sizes (chars): {[int(e.get('prompt_chars') or 0) for e in log[:40]]}"]
    for e in log[:max_entries]:
        parts.append(f"--- sub-call {e.get('i')} · prompt {e.get('prompt_chars')} chars ---\n{str(e.get('prompt') or '')[:chars]}\n--- reply ---\n{str(e.get('response') or '')[:chars // 2]}")
    if len(log) > max_entries:
        parts.append(f"[... {len(log) - max_entries} more sub-calls not shown]")
    return "\n".join(parts)


def render_rollout(row: dict, *, max_chars: int = 30_000, repl_chars: int = 2_500) -> str:
    """Assistant turns + REPL outputs, compact, head+tail truncated."""
    parts: list[str] = []
    k = 0
    for m in row.get("completion") or []:
        role, c = m.get("role"), (m.get("content") or "")
        if role == "assistant":
            k += 1
            parts.append(f"--- assistant turn {k} ---\n{c}")
        elif role == "user":
            parts.append(f"--- REPL/user ---\n{c[:repl_chars]}" + ("\n[...truncated]" if len(c) > repl_chars else ""))
    blob = "\n\n".join(parts)
    if len(blob) > max_chars:
        blob = blob[: max_chars // 2] + f"\n\n[...{len(blob) - max_chars} chars truncated...]\n\n" + blob[-max_chars // 2:]
    return blob


def _task_and_context(row: dict) -> tuple[str, int, str]:
    info = row.get("info") or {}
    task = str(info.get("root_prompt") or "")
    if not task:
        for m in row.get("prompt") or []:
            mt = re.search(r"Answer the following:\s*(.*?)(?:\n\n|$)", m.get("content") or "", re.S)
            if mt:
                task = mt.group(1)
                break
    ctx = info.get("context")
    from rlm_train import doc_store
    if doc_store.is_ref(ctx):
        return task[:4000], doc_store.total_chars(ctx), json.dumps(doc_store.materialize(ctx, limit=3))[:10_000]
    if isinstance(ctx, (list, dict)):
        ctx_text = json.dumps(ctx)[:10_000]
        total = len(json.dumps(ctx))
    else:
        ctx_text = str(ctx or "")[:10_000]
        total = len(str(ctx or ""))
    return task[:4000], total, ctx_text


DERIVED_SFT_ADDENDUM = """
SUB-CALL TRANSCRIPTS. You are also shown the sub-LM calls the root made (prompts abridged, replies shown). The root is
expected to keep its own printed output short: counts, a parse check and a few confirming lines, never the per-item
evidence. Treat a sub-LM reply in the transcripts as visible evidence exactly as if it had been printed: an aggregate
(a count, a ranking, a set of ids, a label distribution) is DERIVED when the transcripts show the sub-LMs labelled,
extracted or judged the items it was computed from, and the root's code visibly parsed those replies and computed the
aggregate from them. Do not call such a rollout LUCKY merely because the root printed the aggregate rather than the
per-item labels. It is still LUCKY if the transcripts do not contain the work (e.g. the sub-calls never saw the
relevant items, their replies do not support the result, or the root's arithmetic does not follow from them).
"""


def render_subcall_evidence(row: dict, *, max_replies: int = 48, reply_chars: int = 420, prompts: int = 3,
                            prompt_chars: int = 700, budget: int = 28_000) -> str:
    """Compact sub-call evidence for the DERIVED / LUCKY check (sizes, prompt heads, first and last replies)."""
    log = row.get("sub_calls") or []
    if not log:
        return "(no sub-calls)"
    n = len(log)
    order = list(range(min(n, max_replies // 2))) + list(range(max(max_replies // 2, n - max_replies // 2), n))
    order = sorted(set(order))
    parts = [f"{n} sub-call(s); prompt sizes (chars): {[int(e.get('prompt_chars') or 0) for e in log[:80]]}"
             + (" ..." if n > 80 else "")]
    for e in log[:prompts]:
        parts.append(f"--- prompt of sub-call {e.get('i')} (head) ---\n{str(e.get('prompt') or '')[:prompt_chars]}")
    used = sum(map(len, parts))
    shown = 0
    for i in order:
        e = log[i]
        piece = f"--- reply {e.get('i')} ---\n{str(e.get('response') or '')[:reply_chars]}"
        if used + len(piece) > budget:
            break
        parts.append(piece)
        used += len(piece)
        shown += 1
    if shown < n:
        parts.append(f"[{n - shown} replies not shown (sampled: the first and last calls)]")
    return "\n".join(parts)


DERIVED_PARTIAL_CREDIT = """
PARTIAL CREDIT. This stream accepts answers with reward >= {floor} (this rollout: {reward}); an answer that differs
from the gold is expected. Judge ONLY whether the submitted answer was computed from visible evidence (sub-call
replies parsed and aggregated by the root's code). Differences from the gold answer (e.g. group sizes, a missing or
extra item) are NOT evidence of luck; LUCKY still applies when the answer does not follow from the visible work.
"""


def derived_messages(row: dict, st: Any = None) -> list[dict]:
    task, total, ctx = _task_and_context(row)
    fa = ""
    for _, b in code_blocks(row.get("completion") or []):
        if 'answer["content"]' in b or "answer['content']" in b:
            fa = b[-600:]
    rollout = render_rollout(row) + "\n\n## SUB-CALL TRANSCRIPTS\n" + render_subcall_evidence(row)
    user = build_derived_prompt(task=task, context_total_len=total, context_first_10k=ctx,
                                rollout_text=rollout, final_answer=fa,
                                gold_answer=str(row.get("answer") or "")[:800])
    if getattr(st, "partial_credit_judge", False):
        user += DERIVED_PARTIAL_CREDIT.format(floor=getattr(st, "min_reward", None), reward=row.get("reward"))
    return [{"role": "system", "content": DERIVED_SYSTEM + DERIVED_SFT_ADDENDUM}, {"role": "user", "content": user}]


def stream_flags_text(st: Any = None, sig: Any = None) -> str:
    """The per-stream acceptance flags and the measured signals the judge should weigh."""
    if st is None:
        return ""
    flags = (f"context_causal={bool(getattr(st, 'context_causal', False))}, "
             f"search_ok={bool(getattr(st, 'search_ok', False))}, python_exact={bool(getattr(st, 'python_exact', False))}")
    if PROBE_FROM_PROMPT_FLAG:
        flags += f", probe_from_prompt_ok={bool(getattr(st, 'probe_from_prompt_ok', False))}"
    meas = ""
    if sig is not None:
        def g(k: str) -> Any:
            return sig.get(k) if isinstance(sig, dict) else getattr(sig, k, None)
        meas = (f"\nmeasured: distinct search patterns={g('n_search_patterns')}, raw-context chars printed="
                f"{g('raw_echo_chars')}, size check before first sub-call={g('size_checked')}, "
                f"batched chunk fan-out={g('chunk_fanout')}, later prompts carry earlier results={g('carries_state')}, "
                f"max sub-prompt chars={g('max_sub_prompt_chars')}")
    note = getattr(st, "judge_note", "") or ""
    return f"## STREAM FLAGS\n{flags}{meas}\n" + (f"intended design for this stream: {note}\n" if note else "") + "\n"


def quality_messages(row: dict, expected_families: tuple[str, ...], hint: str, st: Any = None, sig: Any = None) -> list[dict]:
    task, total, ctx = _task_and_context(row)
    user = (f"## TASK\n{task}\n\n## CONTEXT ({total:,} chars; first 3,000 shown)\n{ctx[:3000]}\n\n"
            f"{stream_flags_text(st, sig)}"
            f"## EXPECTED DECOMPOSITION FAMILIES\n{', '.join(expected_families) or 'unspecified'}\n"
            f"(the model was given this hint, which is NOT part of the saved trajectory: {hint.strip()[:600]!r})\n\n"
            f"## TRAJECTORY\n{render_rollout(row)}\n\n## SUB-CALL TRANSCRIPTS (sample)\n{render_subcalls(row)}\n\n## REWARD\n{row.get('reward')}\n\n"
            f"## GOLD ANSWER\n{str(row.get('answer') or '')[:500]}\n\nReturn the JSON verdict now.")
    return [{"role": "system", "content": QUALITY_SYSTEM}, {"role": "user", "content": user}]


def reverify_quality_messages(row: dict, expected_families: tuple[str, ...], hint: str, st: Any = None,
                              sig: Any = None) -> list[dict]:
    """Like quality_messages, but with each turn's reasoning and the reverify system prompt."""
    from .reverify import render_with_reasoning, reverify_system
    msgs = quality_messages(row, expected_families, hint, st, sig)
    user = msgs[1]["content"].replace(f"## TRAJECTORY\n{render_rollout(row)}\n",
                                      f"## TRAJECTORY\n{render_with_reasoning(row)}\n", 1)
    lf = int(row.get("_reverify_live_from") or 1)
    if lf > 1:
        user = user.replace("Return the JSON verdict now.",
                            f"## REPLAYED PREFIX\nAssistant turns 1-{lf - 1} were replayed verbatim from another rollout (often a "
                            f"weaker policy) and are NOT trained on; the model generated turns {lf} onward. Apply every PROCESS "
                            f"gate (printing, parsing, delegation, verification cycles, reasoning) to turns {lf}+ only, and judge "
                            f"how well they recover from the state the prefix left. In particular checks_independent, "
                            f"no_repeat_after_fail, failure_acknowledged, inventory_before_replan, replan_* and "
                            f"commit_when_verified are about what turns {lf}+ DO: a repeat, a leading check or a weak replan "
                            f"that happens entirely inside turns 1-{lf - 1} does NOT fail them (it is the state being recovered "
                            f"from). The same holds for parse_from_evidence, blind_probe, condensed_output, legible_steps, "
                            f"reads_subcalls_critically, no_redundant_work and the reasoning gates: how the PREFIX first "
                            f"probed, parsed, printed or searched (e.g. no sample-description probe before the first "
                            f"search) is the given state and never fails them; fail them only for what turns {lf}+ do "
                            f"(a turn {lf}+ that parses by guessing, prints raw context, trusts a malformed reply, ...). "
                            f"Grounding, DERIVED and the cycles list still cover the whole trajectory (a check in the "
                            f"prefix that failed counts as a cycle).\n\n"
                            f"Return the JSON verdict now.")
    return [{"role": "system", "content": reverify_system()}, {"role": "user", "content": user}]


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_quality(text: str) -> dict[str, Any]:
    m = _JSON_RE.search(text or "")
    if not m:
        return {"quality": 0, "decomposition": "UNKNOWN", "raw": (text or "")[:300]}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"quality": 0, "decomposition": "UNKNOWN", "raw": (text or "")[:300]}
    d.setdefault("quality", 0)
    return d


def parse_derived(text: str) -> tuple[str, str]:
    mv = re.search(r'"verdict"\s*:\s*"(DERIVED|LUCKY)"', text or "")
    mr = re.search(r'"reason"\s*:\s*"([^"]{0,400})"', text or "")
    return (mv.group(1) if mv else "UNKNOWN", mr.group(1) if mr else (text or "")[:200])


RIGHT_REASON_SYSTEM = (
    "You audit why a model picked an answer. A benchmark gives several candidate replies that differ in ONE attribute; "
    "each option carries a hidden attribute VALUE. The model picked the gold option. Judge ONLY whether the model's "
    "stated reasoning selected it because of the property the gold VALUE names (or something that clearly implies it), "
    "as opposed to an unrelated or opposite property (e.g. gold value 'in_depth' but the model says it chose it for being "
    "'short and punchy').\nReply with exactly one line: `YES | <max 15 words>` or `NO | <max 15 words>` or "
    "`UNCLEAR | <max 15 words>`.")
_FENCE_RE = re.compile(r"```.*?(?:```|$)", re.S)


def _uncheckable_value(v: str) -> bool:
    """A bare boolean or number can't be checked against a stated reason."""
    v = str(v).strip().lower()
    return v in ("true", "false", "") or bool(re.fullmatch(r"[0-9.]+", v))


def right_reason_messages(row: dict) -> list[dict] | None | str:
    """Right-reason prompt, None without option values, or "UNCHECKABLE"."""
    info = row.get("info") or {}
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            return None
    vals = info.get("option_values") if isinstance(info, dict) else None
    gold = str((info.get("correct_letter") if isinstance(info, dict) else None) or row.get("answer") or "").strip().upper()
    if not isinstance(vals, dict) or gold not in vals:
        return None
    if _uncheckable_value(vals[gold]):
        return "UNCHECKABLE"
    prose = [_FENCE_RE.sub("", str(m.get("content") or "")).strip()
             for m in (row.get("completion") or []) if m.get("role") == "assistant"]
    reasoning = "\n---\n".join(p for p in prose[-4:] if p)[-4000:]
    user = ("Options (letter: hidden value):\n" + "\n".join(f"{k}: {v}" for k, v in vals.items())
            + f"\n\nGold = {gold} (value: {vals[gold]})\n\nModel's final reasoning:\n{reasoning}")
    return [{"role": "system", "content": RIGHT_REASON_SYSTEM}, {"role": "user", "content": user}]


def parse_right_reason(text: str) -> tuple[str, str]:
    t = (text or "").strip()
    head = t.split("|")[0].strip().upper()
    for v in ("YES", "NO", "UNCLEAR"):
        if head.startswith(v):
            return v, t[:200]
    return "UNKNOWN", t[:200]


class Judge:
    def __init__(self, cfg: JudgeCfg, concurrency: int = 8):
        self.cfg = cfg
        self.sem = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(timeout=cfg.timeout_s)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _chat(self, messages: list[dict]) -> str:
        headers = {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}
        if self.cfg.headers:
            headers.update(self.cfg.headers)
        body: dict[str, Any] = {"model": self.cfg.model, "messages": messages,
                                "max_tokens": self.cfg.max_tokens, "temperature": 0.0}
        if self.cfg.model.startswith("openai/gpt-5"):
            body["reasoning_effort"] = "minimal"
        async with self.sem:
            for attempt in range(3):
                try:
                    r = await self.client.post(f"{self.cfg.base_url}/chat/completions", json=body, headers=headers)
                    r.raise_for_status()
                    return (r.json()["choices"][0]["message"]["content"] or "").strip()
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        return f"ERROR: {type(e).__name__}: {str(e)[:200]}"
                    await asyncio.sleep(2.0 * (attempt + 1))
        return ""

    async def judge(self, row: dict, *, expected_families: tuple[str, ...], hint: str,
                    run_derived: bool = True, st: Any = None, sig: Any = None, rubric: str = "v1") -> dict[str, Any]:
        q_msgs = (reverify_quality_messages if rubric == "reverify" else quality_messages)(row, expected_families, hint, st, sig)
        q_task = self._chat(q_msgs)
        d_task = self._chat(derived_messages(row, st)) if run_derived else None
        rr_msgs = right_reason_messages(row) if run_derived else None
        rr_task = self._chat(rr_msgs) if isinstance(rr_msgs, list) else None
        q_raw = await q_task
        q = parse_quality(q_raw)
        if q.get("decomposition") == "UNKNOWN":
            q = parse_quality(await self._chat(q_msgs))
        out: dict[str, Any] = {"model": self.cfg.model, "quality": q}
        if d_task is not None:
            v, reason = parse_derived(await d_task)
            out["derived"] = {"verdict": v, "reason": reason}
        if rr_task is not None:
            v, why = parse_right_reason(await rr_task)
            out["right_reason"] = {"verdict": v, "reason": why}
        elif rr_msgs == "UNCHECKABLE":
            out["right_reason"] = {"verdict": "UNCHECKABLE", "reason": "gold value names no property"}
        return out


V2_GATES = ("blind_probe", "parse_from_evidence", "verified_before_submit", "check_independent",
            "reads_subcalls_critically", "legible_steps", "condensed_output")


def accept(judgement: dict, *, min_quality: int = 4, require_derived: bool = True,
           n_sub_calls: int | None = None, min_sub_calls: int = 1, strict: bool = False,
           context_causal: bool = False, rubric: str = "v1", python_exact: bool = False) -> tuple[bool, str]:
    """Accept / reject a verdict; ``strict`` adds extra gates."""
    q = judgement.get("quality") or {}
    if rubric == "reverify":
        from .reverify import accept_reverify
        return accept_reverify(judgement, min_quality=min_quality, require_derived=require_derived,
                               n_sub_calls=n_sub_calls, min_sub_calls=min_sub_calls, context_causal=context_causal,
                               python_exact=python_exact)
    if rubric == "v2":
        strict = True
        for gate in V2_GATES:
            if q.get(gate) is not True:
                return False, f"judge: {gate} = {q.get(gate)}"
    if strict:
        if q.get("search_spam") is not False:
            return False, "judge: guess-and-grep search spam (or unanswered)"
        if q.get("size_checked") is not True:
            return False, "judge: sub-prompt size not checked"
        if context_causal and q.get("causal_ok") is not True:
            return False, "judge: causal context split into independent chunks"
        if n_sub_calls and q.get("concise_subprompts") is not True:
            return False, "judge: sub-prompts not concise"
    if int(q.get("quality") or 0) < min_quality:
        return False, f"quality {q.get('quality')} < {min_quality}"
    zero_call_ok = n_sub_calls == 0 and min_sub_calls == 0
    if not zero_call_ok and not q.get("subcalls_appropriate", False):
        return False, "judge: sub-calls not appropriate"
    if not q.get("grounded", False):
        return False, "judge: answer not grounded"
    if q.get("recovers") is False:
        return False, "judge: repeated errors without recovery"
    if require_derived and (judgement.get("derived") or {}).get("verdict") != "DERIVED":
        return False, f"derived verdict {(judgement.get('derived') or {}).get('verdict')}"
    rr = judgement.get("right_reason")
    if require_derived and rr is not None and rr.get("verdict") != "YES":
        return False, f"judge: right_reason = {rr.get('verdict')}"
    return True, "ok"


def interesting(judgement: dict, *, min_quality: int = 3) -> bool:
    q = judgement.get("quality") or {}
    return bool(q.get("novel")) and int(q.get("quality") or 0) >= min_quality


