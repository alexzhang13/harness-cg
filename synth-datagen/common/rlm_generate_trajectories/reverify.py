"""The "reverify" judge rubric: system prompt, accept / reject rule, mechanical vetoes and reasoning cleanup."""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MAX_K = 5


_CODE_RE = re.compile(r"```repl\s*\n(.*?)```", re.S)
_FENCE_RE = re.compile(r"```.*?(?:```|$)", re.S)
SUBMIT_RE = re.compile(r"answer\s*\[\s*[\"']ready[\"']\s*\]\s*=\s*True")
VERIFY_RE = re.compile(r"\b(verif\w*|double[- ]check\w*|cross[- ]check\w*|sanity[- ]check\w*|confirm\w*|validat\w*|"
                       r"re-?check\w*|check(?:s|ed|er|ing)?)\b", re.I)


def code_of(a: str) -> str:
    return "\n".join(_CODE_RE.findall(a or ""))


def prose_of(a: str) -> str:
    return _FENCE_RE.sub(" ", a or "")


def turns_of(messages: list[dict]) -> list[tuple[dict, str]]:
    """[(assistant message, the REPL output after it)]"""
    out = []
    for i, m in enumerate(messages or []):
        if m.get("role") == "assistant":
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            out.append((m, (nxt.get("content") or "") if nxt and nxt.get("role") == "user" else ""))
    return out


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    sm = difflib.SequenceMatcher(None, a[:3000], b[:3000], autojunk=False)
    return sm.ratio() if sm.quick_ratio() > 0.8 else sm.quick_ratio()


REPEAT_SIM = 0.9
REPEAT_MIN_CHARS = 40
EXACT_MIN_CHARS = 20
_COMMENT_RE = re.compile(r"(?m)^\s*#.*$|\s+#[^\n'\"]*$")


def norm_block(code: str) -> str:
    lines = [ln.rstrip() for ln in _COMMENT_RE.sub("", code or "").splitlines()]
    return "\n".join(ln for ln in lines if ln.strip())


def norm_output(out: str) -> str:
    return re.sub(r"\s+", " ", (out or "").split("REPL variables:")[0])[:600]


REASONING_HINT_RE = re.compile(r"\b(prescribed|as instructed|the hint|the note|strategy hint|per the instructions|"
                               r"the instructions say|the private note|my notes? says?)\b", re.I)
PRIVILEGED_REF_RE = re.compile(r"\b(ground[- ]truth|privileged|the annotations?)\b", re.I)


def reasoning_hint_ref(rc: str, row_hinted: bool = False) -> re.Match | None:
    from .sft_format import _CODE_FENCE_RE
    t = _CODE_FENCE_RE.sub(" ", rc or "")
    return REASONING_HINT_RE.search(t) or (PRIVILEGED_REF_RE.search(t) if row_hinted else None)


def broken_code_turns(messages: list[dict]) -> list[int]:
    """1-based assistant turns whose REPL output has a traceback."""
    T = turns_of([m for m in messages or [] if m.get("role") != "system"])
    return [i + 1 for i, (_, u) in enumerate(T) if "Traceback (most recent call last)" in (u or "")]


NO_RUN_RE = re.compile(r"No REPL output: your message contained no ```repl block|nothing ran", re.I)


def repeated_blocks(turns: list[tuple[dict, str]], *, sim: float = REPEAT_SIM,
                    min_chars: int = REPEAT_MIN_CHARS) -> list[dict]:
    """Assistant turns that repeat an earlier one (retrying a block that never ran doesn't count)."""
    out = []
    ran = [not NO_RUN_RE.search(u or "") for _, u in turns]
    blocks = [norm_block(code_of(m.get("content") or "")) for m, _ in turns]
    texts = [(m.get("content") or "").strip() for m, _ in turns]
    outs = [norm_output(u) for _, u in turns]
    for j in range(1, len(turns)):
        if SUBMIT_RE.search(turns[j][0].get("content") or ""):
            continue
        hit = None
        for i in range(j):
            if not ran[i]:
                continue
            if texts[j] and texts[j] == texts[i]:
                hit = (i, 1.0, "verbatim")
                break
            if len(blocks[j]) >= EXACT_MIN_CHARS and blocks[j] == blocks[i]:
                hit = (i, 1.0, "identical_code")
                break
            if hit is None and len(blocks[j]) >= min_chars and len(blocks[i]) >= min_chars and outs[j] == outs[i]:
                r = _sim(blocks[j], blocks[i])
                if r > sim:
                    hit = (i, r, "near_same_output")
        if hit is not None:
            i, r, kind = hit
            out.append({"turn": j + 1, "of": i + 1, "sim": round(r, 3), "kind": kind, "same_output": outs[j] == outs[i]})
    return out


def mechanical_reasons(row: dict, *, live_from: int = 1, reasoning_needles: tuple[str, ...] = ()) -> list[str]:
    """Mechanical vetoes on a finished row. Turns before ``live_from`` (a replayed prefix) skip the hint check."""
    T = turns_of(row.get("completion") or [])
    reasons: list[str] = []
    if not T:
        return ["no assistant turns"]
    live = T[live_from - 1:]
    for i, (m, _) in enumerate(live):
        rc = m.get("reasoning_content") or ""
        hit = (reasoning_hint_ref(rc, row_hinted=any(n in ROW_HINT_NEEDLES for n in reasoning_needles))
               or next((n for n in reasoning_needles if n and n.lower() in rc.lower()), None))
        if hit:
            what = hit.group(0) if isinstance(hit, re.Match) else hit
            reasons.append(f"reverify: reasoning of turn {i + live_from} refers to the hint/note ({what[:40]!r})")
            break
    reps = repeated_blocks(T)
    if reps:
        r = reps[0]
        reasons.append(f"repeat: turn {r['turn']} repeats turn {r['of']} ({r['kind']}, sim {r['sim']}"
                       + (", same output" if r["same_output"] else "") + f"; {len(reps)} repeated turn(s))")
    tb = [re.sub(r"rlm_worker_\w+\.py", "W", u.split("Traceback", 1)[1][-300:]) for _, u in T if "Traceback" in u]
    if any(tb[i] == tb[i + 1] for i in range(len(tb) - 1)):
        reasons.append("reverify: the same traceback twice in a row")
    sub = next((i for i, (m, _) in enumerate(T) if SUBMIT_RE.search(m.get("content") or "")), None)
    if sub is not None:
        last_check = max((i for i, (m, _) in enumerate(T[:sub + 1]) if VERIFY_RE.search(prose_of(m.get("content") or ""))),
                         default=None)
        if last_check is not None and sub - last_check > 2:
            reasons.append(f"reverify: submit {sub - last_check} turns after the last check (commit when verified)")
    return reasons


RUBRIC_DIR = HERE / "rubric"
REVERIFY_DIR = HERE / "rubric_reverify"
_SHARED = ("00_header.md", "10_decomposition.md", "20_delegation.md", "30_probing_and_parsing.md",
           "40_printing_behavior.md")
_SHARED_TAIL = ("70_grounding_and_overall.md",)


def reverify_system() -> str:
    from .judge_variant import rubric_text
    parts = [rubric_text(RUBRIC_DIR / f).rstrip("\n") for f in _SHARED]
    parts += [p.read_text().rstrip("\n") for p in sorted(REVERIFY_DIR.glob("[0-6]*.md"))]
    parts += [(RUBRIC_DIR / f).read_text().rstrip("\n") for f in _SHARED_TAIL]
    parts.append((REVERIFY_DIR / "90_response_format.md").read_text().rstrip("\n"))
    return "\n".join(parts) + "\n"


def render_with_reasoning(row: dict, *, max_chars: int = 40_000, repl_chars: int = 2_500, reasoning_chars: int = 1_800) -> str:
    """judge.render_rollout plus each turn's clipped reasoning."""
    parts: list[str] = []
    k = 0
    for m in row.get("completion") or []:
        role, c = m.get("role"), (m.get("content") or "")
        if role == "assistant":
            k += 1
            rc = (m.get("reasoning_content") or "").strip()
            if len(rc) > reasoning_chars:
                rc = rc[: reasoning_chars // 2] + f"\n[... {len(rc) - reasoning_chars} chars ...]\n" + rc[-reasoning_chars // 2:]
            parts.append(f"--- assistant turn {k} ---\n" + (f"[reasoning]\n{rc}\n[/reasoning]\n" if rc else "[no reasoning]\n") + c)
        elif role == "user":
            parts.append(f"--- REPL/user ---\n{c[:repl_chars]}" + ("\n[...truncated]" if len(c) > repl_chars else ""))
    blob = "\n\n".join(parts)
    if len(blob) > max_chars:
        blob = blob[: max_chars // 2] + f"\n\n[...{len(blob) - max_chars} chars truncated...]\n\n" + blob[-max_chars // 2:]
    return blob


REVERIFY_GATES = ("parse_from_evidence", "reads_subcalls_critically", "condensed_output", "legible_steps", "recovers",
                  "checks_independent", "failure_acknowledged", "no_repeat_after_fail", "inventory_before_replan",
                  "replan_from_information", "replan_is_new", "reverified_before_submit", "commit_when_verified",
                  "reasoning_present", "reasoning_consistent", "reasoning_clean", "reasoning_decides",
                  "no_redundant_work")


def cycles_consistent(q: dict) -> str:
    """'' if the judge's per-cycle annotation is coherent and acceptable, else why not."""
    cyc = q.get("cycles")
    if not isinstance(cyc, list):
        return "cycles missing"
    fails = [c for c in cyc if isinstance(c, dict) and c.get("result") == "fail"]
    try:
        k = int(q.get("n_failed_checks"))
    except (TypeError, ValueError):
        return "n_failed_checks missing"
    if k != len(fails):
        return f"n_failed_checks {k} != {len(fails)} failed cycles"
    if k > MAX_K:
        return f"{k} failed checks > {MAX_K}"
    if not cyc:
        return "no check at all"
    for c in cyc:
        if c.get("independent") is not True:
            return f"check at turn {c.get('check_turn')} not independent"
        if c.get("read_correctly") is not True:
            return f"check at turn {c.get('check_turn')} misread"
    for c in fails:
        if c.get("replan_turn") is None or c.get("replan_grounded") is not True or c.get("replan_new") is not True:
            return f"failed check at turn {c.get('check_turn')}: replan missing / ungrounded / not new"
    if cyc[-1].get("result") != "pass" and q.get("reverified_before_submit") is not True:
        return "last check did not pass"
    return ""


def accept_reverify(judgement: dict, *, min_quality: int = 4, require_derived: bool = True,
                    n_sub_calls: int | None = None, min_sub_calls: int = 1, context_causal: bool = False,
                    python_exact: bool = False) -> tuple[bool, str]:
    q = judgement.get("quality") or {}
    if python_exact and q.get("programmatic_rationale") is False:
        return False, "judge: programmatic_rationale = False"
    for gate in REVERIFY_GATES:
        if q.get(gate) is not True:
            return False, f"judge: {gate} = {q.get(gate)}"
    if q.get("false_pass") is not False:
        return False, f"judge: false_pass = {q.get('false_pass')}"
    why = cycles_consistent(q)
    if why:
        return False, f"judge: cycles: {why}"
    if q.get("search_spam") is not False:
        return False, "judge: guess-and-grep search spam (or unanswered)"
    if q.get("size_checked") is not True:
        return False, "judge: sub-prompt size not checked"
    if context_causal and q.get("causal_ok") is not True:
        return False, "judge: causal context split into independent chunks"
    if int(q.get("quality") or 0) < min_quality:
        return False, f"quality {q.get('quality')} < {min_quality}"
    zero_call_ok = n_sub_calls == 0 and min_sub_calls == 0
    if not zero_call_ok and not q.get("subcalls_appropriate", False):
        return False, "judge: sub-calls not appropriate"
    if not q.get("grounded", False):
        return False, "judge: answer not grounded"
    if require_derived and (judgement.get("derived") or {}).get("verdict") != "DERIVED":
        return False, f"derived verdict {(judgement.get('derived') or {}).get('verdict')}"
    rr = judgement.get("right_reason")
    if require_derived and rr is not None and rr.get("verdict") != "YES":
        return False, f"judge: right_reason = {rr.get('verdict')}"
    return True, "ok"


def reverify_meta(judgement: dict) -> dict[str, Any]:
    """meta.reverify for an accepted row: k, cycles and the error turns."""
    q = judgement.get("quality") or {}
    cyc = [c for c in (q.get("cycles") or []) if isinstance(c, dict)]
    err = sorted({int(t) for c in cyc if c.get("result") == "fail" for t in (c.get("error_turns") or []) if str(t).isdigit()})
    return {"k": int(q.get("n_failed_checks") or 0), "cycles": cyc, "error_turns": err}


REWRITE_SYSTEM = """You edit the private reasoning of one turn of an agent that works in a Python REPL. The agent was
given a private note or instruction that will be REMOVED from the training transcript, and its reasoning mentions it.
Rewrite the reasoning so that it never refers to any note, hint, instruction, guidance, template, strategy, plan or
procedure it was given ("the plan says", "following the plan", "step 5"), turn budget, earlier attempt, grader or
expected answer. Keep every observation, decision and plan, in the same order and at about the same
length; where the original justifies a step by the note ("the note says to re-check"), justify it instead by what the
outputs shown so far establish ("the count disagreed, so ..."). Do not add facts, results or conclusions that are not
in the original or in the visible text. Output ONLY the rewritten reasoning."""


SCAFFOLD_RE = re.compile(
    r"\b(?:(?:follow(?:ing|s)?|per|according to|as in|as (?:per|shown in)|stick(?:ing)? to) (?:the|my|our) (?:plan|procedure|"
    r"steps|instructions|guidance|guidelines|rules|recipe|checklist)|the (?:plan|procedure|instructions?|guidance|rules|recipe|"
    r"checklist) (?:says|said|tells?|told|requires?|wants?|asks?|calls? for)|(?:turn|step) budget|budget of \d+|"
    r"step \d+(?:\([a-d]\))? (?:of|in) the (?:plan|procedure)|the (?:grader|scaffold\w*|system prompt)|"
    r"(?:my|the) (?:strategy|instructions) (?:says|say|said))\b", re.I)


async def sanitize_reasoning(judge: Any, row: dict, *, needles: tuple[str, ...] = (), live_from: int = 1) -> int:
    """Rewrite in place the reasoning of turns that mention the hint. Returns the number of turns rewritten."""
    comp = row.get("completion") or []
    seen, todo = 0, []
    for i, m in enumerate(comp):
        if m.get("role") != "assistant":
            continue
        seen += 1
        rc = m.get("reasoning_content") or ""
        if seen >= live_from and rc and (reasoning_hint_ref(rc, row_hinted=any(n in ROW_HINT_NEEDLES for n in needles))
                                         or SCAFFOLD_RE.search(rc)
                                         or any(n and n.lower() in rc.lower() for n in needles)):
            prev = comp[i - 1]["content"] if i > 0 and comp[i - 1].get("role") == "user" else ""
            todo.append((i, rc, m.get("content") or "", prev or ""))
    if not todo:
        return 0

    async def one(i: int, rc: str, content: str, prev: str) -> None:
        user = (f"## LAST OUTPUT THE AGENT SAW (truncated)\n{prev[:3000]}\n\n## THE TURN'S VISIBLE TEXT\n{content[:3000]}\n\n"
                f"## REASONING TO REWRITE\n{rc}")
        out = await judge._chat([{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": user}])
        if out and not out.startswith("ERROR:"):
            comp[i] = {**comp[i], "reasoning_content": out.strip(), "reasoning_rewritten": True}

    import asyncio
    await asyncio.gather(*[one(*t) for t in todo])
    return sum(1 for i, *_ in todo if comp[i].get("reasoning_rewritten"))


ROW_HINT_NEEDLES = ("PRIVILEGED GROUND TRUTH", "benchmark's own annotations")


