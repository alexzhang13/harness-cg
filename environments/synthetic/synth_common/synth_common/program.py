"""Hints for the root (``HintSpec``) and program-level checks on finished rollouts (``Check``).

Checks are logged as ``prog_<name>`` and combined into ``program_score``. ``reward_mode``: ``outcome`` (checks are
logged only), ``gated`` (outcome scaled by ``gate_floor`` unless every hard check passes) or ``shaped`` (outcome mixed
with program_score).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

HINT_OPEN, HINT_CLOSE = "<hint>", "</hint>"
_HINT_RE = re.compile(r"\n*<hint>.*?</hint>\n*", re.S)


VERIFY_COMMON = """Before you submit, check your answer with the sub-LM. This step is required, even when you are confident:
1. Finish the answer in Python first (keep it in a variable), then pick the evidence it rests on: the few passages, table windows or functions your result came from -- a few thousand characters each, never the whole context.
2. Send them with llm_query / llm_query_batched and a NEUTRAL question: ask for the fact, value or verdict and a verbatim quote. Never put your answer, or the value you expect, in the prompt.
3. In Python, confirm each quote really occurs in the passage you sent, then compare the replies with your answer.
4. If they agree, submit in the next block. If they disagree, say what disagreed, re-derive only that part (a narrower window, a missing passage, the case your parser missed) and check again. Never re-send the same check unchanged.
Arithmetic, counting, sorting and parsing stay in Python; the sub-LM only reads and quotes."""


@dataclass
class HintSpec:
    strategy: str = ""
    program: str = ""
    verify: str = ""
    verify_base: str = ""

    def text(self, tier: str) -> str:
        if tier in ("none", "", None):
            return ""
        if tier == "strategy":
            return self.strategy.strip()
        if tier == "verify":
            body = (self.verify_base or self.strategy).strip() + "\n\n" + VERIFY_COMMON
            if self.verify.strip():
                body += "\n\nWhat to check on this task: " + self.verify.strip()
            return body
        if tier == "program":
            body = self.strategy.strip()
            if self.program.strip():
                # ```repl, not ```python: the harness only executes repl blocks
                body += "\n\nA program that does this (adapt it; run it as a few ```repl blocks, one job per block " \
                        "-- parse, compute, check, submit -- each printing a compact summary):\n```repl\n" + \
                        self.program.strip("\n") + "\n```"
            return body
        raise ValueError(f"unknown hint tier {tier!r}; expected none | strategy | program | verify")


FORMAT_RULE = ("Format: every piece of code goes inside a ```repl fenced block. Do not wrap code in <tool_call>, "
               "<function=...> or <parameter=...> tags: such a turn executes nothing and costs a turn.")


def add_hint(root_prompt: str, hint_text: str) -> str:
    if not hint_text:
        return root_prompt
    return f"{root_prompt}\n\n{HINT_OPEN}\n{hint_text}\n\n{FORMAT_RULE}\n{HINT_CLOSE}"


def strip_hint(text: str) -> str:
    """Remove an injected hint block (for hint-drop / privileged-information training)."""
    return _HINT_RE.sub("\n", text).rstrip()


@dataclass
class Check:
    name: str
    fn: Callable[[dict], float | None]
    hard: bool = False
    weight: float = 1.0
    doc: str = ""


def acct(state: dict) -> dict:
    return state.get("synth") or {}


def required_coverage(state: dict) -> float:
    req = (state.get("synth_hidden") or {}).get("required_ids") or []
    if not req:
        return 1.0
    read = acct(state).get("read_ids") or {}
    return sum(1 for r in req if r in read) / len(req)


def root_code(state: dict) -> list[str]:
    """The ```repl blocks the root wrote, in order."""
    out = []
    for m in (state.get("rlm_history") or [])[2:]:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role != "assistant":
            continue
        c = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        out.extend(re.findall(r"```repl\s*\n(.*?)```", c or "", re.S))
    return out


def sub_responses(state: dict) -> list[str]:
    return [e.get("response") or "" for e in (state.get("rlm_sub_llm_log") or [])]


def final_answer(state: dict) -> str:
    return str(state.get("rlm_final_answer") or "")


def _norm(s: str) -> str:
    return re.sub(r"[\s`*_\"']+", " ", s).strip().lower()


def coverage(min_cov: float = 0.95, hard: bool = True) -> Check:
    return Check("coverage", lambda s: min(1.0, required_coverage(s) / min_cov), hard=hard,
                 doc=f"read >= {min_cov:.0%} of the items a correct solution needs")


def no_overlong(max_n: int = 0, hard: bool = False) -> Check:
    return Check("no_overlong", lambda s: 1.0 if acct(s).get("overlong", 0) <= max_n else 0.0, hard=hard,
                 doc="no sub-call over the sub-LM's window")


def no_aggregate_requests(max_n: int = 0, hard: bool = False) -> Check:
    return Check("no_agg_requests", lambda s: 1.0 if acct(s).get("agg_requests", 0) <= max_n else 0.0, hard=hard,
                 doc="never asked the sub-LM for a count/ranking/answer over more items than it can hold")


def recovered_skips(hard: bool = False) -> Check:
    """Items the oracle skipped under load were re-asked until the required set was covered."""
    def fn(s: dict) -> float:
        if acct(s).get("items_skipped", 0) == 0:
            return 1.0
        return 1.0 if required_coverage(s) >= 0.999 else 0.0
    return Check("recovered_skips", fn, hard=hard, doc="re-asked whatever the sub-LM silently dropped")


def answer_not_copied(min_len: int = 1, hard: bool = False) -> Check:
    """The submitted answer is not simply one sub-call's reply (the reduction happened in Python)."""
    def fn(s: dict) -> float:
        fa = _norm(final_answer(s))
        if len(fa) < min_len:
            return 1.0
        return 0.0 if any(_norm(r) == fa for r in sub_responses(s)) else 1.0
    return Check("answer_not_copied", fn, hard=hard, doc="final answer computed, not pasted from a sub-call")


def max_turns(n: int, hard: bool = False) -> Check:
    return Check("turns", lambda s: 1.0 if int(s.get("rlm_iterations") or 0) <= n else 0.0, hard=hard,
                 doc=f"done within {n} root turns")


def used_batched(min_batches: int = 1, hard: bool = False) -> Check:
    return Check("batched", lambda s: 1.0 if acct(s).get("batches", 0) >= min_batches else 0.0, hard=hard,
                 doc="fanned out with llm_query_batched")


def code_mentions(name: str, pattern: str, hard: bool = False, doc: str = "") -> Check:
    rx = re.compile(pattern, re.S)
    return Check(name, lambda s: 1.0 if any(rx.search(c) for c in root_code(s)) else 0.0, hard=hard, doc=doc)


_ANSWER_PREFIX = re.compile(r"^(the )?(correct |final )?answer( is)?[:\s]*", re.I)


def answer_core(text: str) -> str:
    """The submitted value without its boilerplate ("The correct answer is ..."), normalised."""
    core = _ANSWER_PREFIX.sub("", str(text or "").strip())
    return _norm(core).strip("().[] ")


def call_log(state: dict) -> list[dict]:
    return acct(state).get("call_log") or []


def delegation_stats(calls: list[dict], answer: str, iterations: int) -> dict[str, float]:
    """(ctx_calls, late_check, late_neutral) from sub-call records: context-carrying calls in the last two turns."""
    core = answer_core(answer)
    leak = re.compile(r"(?<![a-z0-9])" + re.escape(core) + r"(?![a-z0-9])") if len(core) >= 2 else None
    last = max(1, int(iterations or 0))
    late = [c for c in calls if c.get("ctx") and int(c.get("step") or 0) >= last - 1]
    neutral = [c for c in late if leak is None or not leak.search(_norm(c.get("residual") or ""))]
    return {"ctx_calls": float(sum(1 for c in calls if c.get("ctx"))), "late_check": float(bool(late)),
            "late_neutral": float(bool(neutral))}


def delegated(min_calls: int = 1, hard: bool = False) -> Check:
    return Check("delegated", lambda s: 1.0 if sum(1 for c in call_log(s) if c.get("ctx")) >= min_calls else 0.0,
                 hard=hard, doc=f">= {min_calls} sub-call(s) carrying text from the context")


def verified_late(hard: bool = False) -> Check:
    """Re-verify: a context-carrying, neutral sub-call in the submit turn or the one before it."""
    def fn(s: dict) -> float | None:
        if not final_answer(s):
            return None
        st = delegation_stats(call_log(s), final_answer(s), int(s.get("rlm_iterations") or 0))
        return st["late_neutral"]
    return Check("verified_late", fn, hard=hard,
                 doc="checked the answer with a neutral, evidence-carrying sub-call right before submitting")


def evaluate(checks: list[Check], state: dict) -> dict[str, float | None]:
    """name -> score in [0, 1], or None if the check doesn't apply (excluded from program_score and gates)."""
    out: dict[str, float | None] = {}
    for c in checks:
        try:
            v = c.fn(state)
            out[c.name] = None if v is None else float(max(0.0, min(1.0, v)))
        except Exception:  # noqa: BLE001  broken check counts as failed
            out[c.name] = 0.0
    return out


def program_score(checks: list[Check], vals: dict[str, float | None]) -> float:
    """Weighted mean over the checks that apply (1.0 if none apply)."""
    applicable = [c for c in checks if vals.get(c.name) is not None]
    tw = sum(c.weight for c in applicable)
    if tw <= 0:
        return 1.0
    return sum(c.weight * float(vals[c.name]) for c in applicable) / tw


def combine(outcome: float, checks: list[Check], vals: dict[str, float], mode: str, program_weight: float = 0.2,
            gate_floor: float = 0.0) -> float:
    if mode == "outcome" or not checks:
        return outcome
    if mode == "gated":
        ok = all(vals.get(c.name) is None or vals[c.name] >= 0.999 for c in checks if c.hard)
        return outcome if ok else gate_floor * outcome
    if mode == "shaped":
        ps = program_score(checks, vals)
        return (1.0 - program_weight) * outcome + program_weight * ps * (1.0 if outcome > 0 else 0.0)
    raise ValueError(f"unknown reward_mode {mode!r}; expected outcome | gated | shaped")


__all__ = [
    "HINT_CLOSE", "HINT_OPEN", "VERIFY_COMMON", "Check", "HintSpec", "add_hint", "answer_core", "answer_not_copied",
    "call_log", "code_mentions", "combine", "coverage", "delegated", "delegation_stats", "evaluate", "final_answer",
    "max_turns", "no_aggregate_requests", "no_overlong", "program_score", "recovered_skips", "required_coverage",
    "root_code", "strip_hint", "sub_responses", "used_batched", "verified_late",
]
