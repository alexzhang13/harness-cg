"""Filler pool (DCLM chunks, ``$LONGREASON_FILLER``) and context assembly at LongReason's lengths."""

from __future__ import annotations

import gzip
import json
import os
import random
from functools import lru_cache
from pathlib import Path
from rlm_train.frozen import frozen_path

POOL_PATH = Path(os.environ.get("LONGREASON_FILLER") or Path(__file__).parent / "data" / "filler.jsonl.gz")

# median len(background) per LongReason split, in chars
LENGTH_CHARS = {"8k": 23_000, "16k": 54_000, "32k": 114_000, "64k": 235_000, "128k": 474_000}


@lru_cache(maxsize=1)
def pool() -> tuple[str, ...]:
    with gzip.open(frozen_path(POOL_PATH, "longreason/filler.jsonl.gz"), "rt", encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f)


def sample_filler(rng: random.Random, n_chars: int) -> list[str]:
    """Distinct chunks totalling about ``n_chars`` (no chunk repeats inside one context)."""
    p = pool()
    out, tot = [], 0
    for i in rng.sample(range(len(p)), k=min(len(p), n_chars // 250 + 64)):
        out.append(p[i])
        tot += len(p[i]) + 2
        if tot >= n_chars:
            break
    return out


def assemble(rng: random.Random, filler: list[str], scenario: list[str], distractors: list[str]) -> str:
    """Spread scenario paragraphs in order over the context (as LongReason does); distractors go anywhere."""
    n = len(filler)
    slots = sorted(rng.sample(range(n + 1), k=min(len(scenario), n + 1)))
    ins: dict[int, list[str]] = {}
    for s, para in zip(slots, scenario, strict=False):
        ins.setdefault(s, []).append(para)
    for para in distractors:
        ins.setdefault(rng.randrange(n + 1), []).insert(0, para)
    parts: list[str] = []
    for i in range(n + 1):
        parts.extend(ins.get(i, ()))
        if i < n:
            parts.append(filler[i])
    return "\n\n".join(parts)


__all__ = ["LENGTH_CHARS", "assemble", "pool", "sample_filler"]
