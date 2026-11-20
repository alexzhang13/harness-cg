"""The SFT-row validity check applied to every accepted row at ingestion and again at export."""

from __future__ import annotations

import json

from .filters import THINK_LEAK_RE
from .strategies import HINT_CLOSE, HINT_OPEN


def validate(row: dict) -> str | None:
    """None if the row is a usable SFT sample, else why it is not."""
    msgs = row.get("messages") or []
    if not msgs:
        return "no messages"
    blob = json.dumps([{"role": m.get("role"), "content": m.get("content")} for m in msgs])
    if HINT_OPEN in blob or HINT_CLOSE in blob or "strategy_hint" in blob:
        return "hint text leaked into the transcript"
    if THINK_LEAK_RE.search(blob):
        return "thinking or tool-call tags in the transcript"
    if msgs[0].get("role") != "system":
        return f"first message is {msgs[0].get('role')!r}, not system"
    if not any(m.get("role") == "assistant" and (m.get("content") or "").strip() for m in msgs):
        return "no non-empty assistant turn"
    if any(m.get("role") == "assistant" and (m.get("content") or "").startswith("\n") for m in msgs):
        return "an assistant turn starts with a newline"
    roles = [m.get("role") for m in msgs]
    if roles.count("user") < 2:
        return f"prompt layout looks wrong: {roles[:4]}"
    return None

