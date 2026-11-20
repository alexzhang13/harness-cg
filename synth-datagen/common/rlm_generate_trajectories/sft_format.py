"""Turn a conditioned rollout into a standard RLM SFT sample."""

from __future__ import annotations

import re

from typing import Any

from .strategies import HINT_OPEN, HINT_CLOSE, strip_hint_text


class HintLeak(ValueError):
    """The hint text survived stripping somewhere in the sample."""


def _plain(m: dict) -> dict:
    """role/content, plus the model's ``reasoning_content`` when it has any."""
    c = m.get("content") or ""
    if m.get("role") == "assistant":
        c = c.strip("\n")
    out = {"role": m.get("role"), "content": c}
    rc = m.get("reasoning_content")
    if m.get("role") == "assistant" and isinstance(rc, str) and rc.strip():
        out["reasoning_content"] = rc.strip("\n")
    return out


def strip_hint(prompt: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in prompt:
        c = m.get("content") or ""
        if m.get("role") == "user" and HINT_OPEN in c:
            rest = strip_hint_text(c)
            if rest:
                out.append({"role": "user", "content": rest})
            continue
        out.append(_plain(m))
    return out


HINT_REFERENCE_RE = re.compile(
    r"\b(prescribed|as instructed|as specified|the specified|the hint|the note|the template|strategy hint|"
    r"expected (?:layout|format|structure)|per the instructions|the instructions say)\b", re.I)
_CODE_FENCE_RE = re.compile(r"```.*?(?:```|$)", re.S)


_TASK_GROUNDED = (("template", "templat"), ("specified", "specif"), ("expected", "format"))


def task_text_of(messages: list[dict]) -> str:
    """The user-visible task: user messages before the first assistant turn (the hint is already stripped)."""
    out = []
    for m in messages or []:
        if m.get("role") == "assistant":
            break
        if m.get("role") == "user":
            out.append(str(m.get("content") or ""))
    return "\n".join(out).lower()


def _ref(text: str, task_text: str = "") -> re.Match | None:
    for x in HINT_REFERENCE_RE.finditer(text):
        hit = x.group(0).lower()
        if task_text and any(w in hit and stem in task_text for w, stem in _TASK_GROUNDED):
            continue
        return x
    return None


def hint_reference(content: str, task_text: str = "") -> re.Match | None:
    """Does assistant prose (not code) refer to the stripped hint? Phrases the task text itself uses are excused."""
    return _ref(_CODE_FENCE_RE.sub(" ", content), task_text)


def to_sft_sample(row: dict, *, strategy_name: str, hint: str, judge: dict | None = None,
                  extra_meta: dict | None = None, extra_needles: tuple[str, ...] = ()) -> dict[str, Any]:
    """Build the SFT row. Raises HintLeak if any message still contains the hint."""
    prompt = strip_hint(list(row.get("prompt") or []))
    completion = [_plain(m) for m in (row.get("completion") or [])]
    messages = prompt + completion
    roles = [m["role"] for m in prompt]
    if not (roles[:1] == ["system"] and roles.count("user") >= 2):
        raise ValueError(f"unexpected prompt layout after stripping: {roles}")
    needle = hint.strip()[:80]
    extra = [n for n in (x.strip() for x in extra_needles) if n]
    task_text = task_text_of(prompt)
    for m in messages:
        c = m["content"]
        if HINT_OPEN in c or HINT_CLOSE in c or (needle and needle in c):
            raise HintLeak("hint text survived in a message")
        if any(n in c for n in extra):
            raise HintLeak("replay-hint text survived in a message")
        ref = hint_reference(c, task_text) if m.get("role") == "assistant" else None
        if ref:
            raise HintLeak(f"assistant prose refers to the stripped hint ({ref.group(0)!r})")
    reasoning_leak = None
    for m in completion:
        r = m.get("reasoning_content") or ""
        if r and (HINT_OPEN in r or HINT_CLOSE in r or (needle and needle in r) or any(n in r for n in extra)):
            reasoning_leak = "hint text in reasoning"
        elif r and _ref(r, task_text):
            reasoning_leak = reasoning_leak or f"reasoning refers to the hint ({_ref(r, task_text).group(0)!r})"
    meta = {
        "reasoning_turns": sum(1 for m in completion if m.get("reasoning_content")),
        "reasoning_hint_leak": reasoning_leak,
        "strategy": strategy_name,
        "example_id": row.get("example_id"),
        "reward": row.get("reward"),
        "rlm_sub_llm_calls": row.get("rlm_sub_llm_calls", (row.get("metrics") or {}).get("rlm_sub_llm_calls")),
        "rlm_iterations": row.get("rlm_iterations", (row.get("metrics") or {}).get("rlm_iterations")),
        "stop_condition": row.get("stop_condition"),
        "sub_prompt_chars": row.get("rlm_sub_llm_prompt_chars"),
        "sub_response_chars": row.get("rlm_sub_llm_response_chars"),
        "judge": judge,
    }
    if extra_meta:
        meta.update(extra_meta)
    return {"messages": messages, "prompt": prompt, "completion": completion, "meta": meta}
