"""Task identity."""

from __future__ import annotations

import hashlib
import re
from typing import Any

_CTX_RE = re.compile(r"Your context is a (\w+) of ([\d,]+) total characters")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def metadata_message(messages: list[dict] | None) -> str | None:
    """The metadata user message of a saved row (first user message after the system prompt)."""
    for m in (messages or [])[1:3]:
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content") or "")
    return None


def meta_sha1_of_messages(messages: list[dict] | None) -> str | None:
    t = metadata_message(messages)
    return sha1(t) if t else None


def meta_sha1_of_info(info: Any) -> str:
    from ..replay import metadata_text
    return sha1(metadata_text(info))


def task_sha1_of_info(info: Any) -> str:
    from ..replay import task_sha1
    return task_sha1(info)


def context_sha1_of_info(info: Any) -> str:
    from ..replay import _ctx_str
    return sha1(_ctx_str(info))


def task_key(task_sha1: str | None, meta_sha1: str | None) -> str | None:
    if task_sha1:
        return "c:" + task_sha1
    if meta_sha1:
        return "m:" + meta_sha1
    return None


def group_key(count_by: str, task_sha1: str | None, meta_sha1: str | None) -> str | None:
    """The key targets are counted by."""
    if count_by == "task":
        return task_key(task_sha1, meta_sha1)
    return ("m:" + meta_sha1) if meta_sha1 else task_key(task_sha1, None)


def question_snippet(meta_text: str | None, n: int = 240) -> str:
    """Short human label for a task: the question part of the metadata message."""
    t = (meta_text or "").strip()
    if t.startswith("Answer the following:"):
        t = t[len("Answer the following:"):].strip()
    i = t.find("Your context is a")
    if i > 0:
        t = t[:i].strip()
    t = re.sub(r"\s+", " ", t)
    return t if len(t) <= n else t[: n - 1] + "…"


def context_chars(meta_text: str | None) -> int | None:
    m = _CTX_RE.search(meta_text or "")
    return int(m.group(2).replace(",", "")) if m else None
