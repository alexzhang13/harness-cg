"""Privilege-count split logic for the ManyIH-Bench instruction-following env."""

from __future__ import annotations

import re
from typing import Any

_TAG_RE = re.compile(r"\[\[Privilege\s+\d+\]\]")

TIERS: list[tuple[int, str, int, int]] = [
    (1, "minimal", 2, 4),
    (2, "low", 5, 7),
    (3, "moderate", 8, 10),
    (4, "high", 11, 14),
    (5, "extreme", 15, 10_000),
]


def render_messages(messages: list[dict[str, str]]) -> str:
    """Flatten the dataset's chat-format input into a single context string."""
    parts: list[str] = []
    for m in messages:
        role = (m.get("role") or "").upper()
        content = m.get("content") or ""
        parts.append(f"### {role}\n{content}")
    return "\n\n".join(parts)


def _render(input_msgs: list[dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content") or "") for m in input_msgs)


def count_tagged_directives(input_msgs: list[dict[str, Any]]) -> int:
    """Number of ``[[Privilege N]]`` tags across all messages of one task."""
    return len(_TAG_RE.findall(_render(input_msgs)))


def tier_of(n_tags: int) -> int:
    """Map a tag count to its tier id (1-5); 0 if below the minimum (1 tag)."""
    for tier_id, _label, lo, hi in TIERS:
        if lo <= n_tags <= hi:
            return tier_id
    return 0


def _in_split(
    n_tags: int,
    split: str,
    train_max_tags: int,
    eval_min_tags: int,
) -> bool:
    if split == "train":
        return n_tags <= train_max_tags
    if split == "eval":
        return n_tags >= eval_min_tags
    raise ValueError(f"split={split!r} must be 'train' or 'eval'.")


def select_rows(
    rows: list[dict[str, Any]],
    *,
    split: str = "train",
    train_max_tags: int = 5,
    eval_min_tags: int = 15,
) -> list[dict[str, Any]]:
    """Filter dataset rows by tag-count split."""
    out: list[dict[str, Any]] = []
    for ex in rows:
        n_tags = count_tagged_directives(list(ex.get("input") or []))
        t = tier_of(n_tags)
        if not _in_split(n_tags, split, train_max_tags, eval_min_tags):
            continue
        ex = dict(ex)
        ex["_n_tagged"] = n_tags
        ex["_tier"] = t
        out.append(ex)
    return out


__all__ = [
    "TIERS",
    "count_tagged_directives",
    "tier_of",
    "select_rows",
    "render_messages",
]
