"""Append <train dir>/../train_extra/<task>/train.jsonl (the seed-43 build of build_extra.sh) to
<train dir>/<task>/train.jsonl: old rows first, duplicates (same queries + documents) dropped, a copy of the
original kept as train.jsonl.bak300. Usage: python merge_extra.py <task> [<task> ...]"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

TRAIN = Path(os.environ.get("CTCBENCH_TRAIN_DIR") or Path.home() / "ctcbench_data" / "train")


def key(row: dict) -> str:
    docs = "\x1e".join(d.get("text", "") for d in row.get("documents") or [])
    return hashlib.sha1((json.dumps(row.get("queries"), sort_keys=True) + "\x1d" + docs).encode()).hexdigest()


for task in sys.argv[1:]:
    old_p, new_p = TRAIN / task / "train.jsonl", TRAIN.parent / "train_extra" / task / "train.jsonl"
    if not new_p.exists():
        print(f"{task}: no extra build, skipped")
        continue
    old = [l for l in open(old_p) if l.strip()]
    seen = {key(json.loads(l)) for l in old}
    add = []
    for l in open(new_p):
        if l.strip() and (k := key(json.loads(l))) not in seen:
            seen.add(k)
            add.append(l)
    bak = old_p.with_suffix(".jsonl.bak300")
    if not bak.exists():
        shutil.copy(old_p, bak)
    with open(old_p, "w") as f:
        f.writelines(old + add)
    print(f"{task}: {len(old)} + {len(add)} = {len(old) + len(add)}")
