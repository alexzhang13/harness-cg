"""Build the filler pool from one DCLM-baseline shard (used instead of LongReason's filler to avoid eval text).

    python -m longreason_expanded.build_filler [--n 9000] [--shard <dclm parquet shard>]
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import random
import re
from pathlib import Path

from .filler import POOL_PATH

OUT = POOL_PATH
_BAD = re.compile(r"https?://|www\.|[{}<>|@#\\]|\.(?:com|org|net)\b|^\W*$|cookie|javascript|subscribe", re.I)


def _prose(s: str) -> bool:
    if len(s) < 40 or _BAD.search(s):
        return False
    letters = sum(ch.isalpha() for ch in s)
    return letters / len(s) >= 0.72 and s.count(" ") >= 6 and s.isascii()


def build(shard: str, n: int, seed: int = 0) -> list[str]:
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    texts = pq.read_table(shard, columns=["text"]).column("text").to_pylist()
    rng.shuffle(texts)
    chunks, seen = [], set()
    for t in texts:
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        i = 0
        while i < len(lines) and len(chunks) < n:
            k = rng.choice([1, 1, 1, 2, 2, 3, 4, 6])
            block = [ln for ln in lines[i:i + k] if _prose(ln)]
            i += k
            if not block:
                continue
            c = "\n".join(block)
            if not 120 <= len(c) <= 1500 or c in seen:
                continue
            seen.add(c)
            chunks.append(c)
            if rng.random() < 0.5:          # ~2 chunks per document
                break
        if len(chunks) >= n:
            break
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9000)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args()
    shard = a.shard or sorted(glob.glob(
        str(Path.home() / ".cache/huggingface/hub/datasets--mlfoundations--dclm-baseline-1.0-parquet/snapshots/*/**/*.parquet"),
        recursive=True))[0]
    chunks = build(shard, a.n)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    tot = sum(len(c) for c in chunks)
    print(f"wrote {len(chunks)} chunks, {tot:,} chars ({tot / len(chunks):.0f} avg) -> {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
