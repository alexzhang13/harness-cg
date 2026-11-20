"""Freeze CTC-Bench split v1 into ctcbench_real/split_v1.json (run once; the env only reads the file).

Drops every train row that shares a source item (gold document, context document, book / source id) with a test row
of its own task or a sibling task built from the same seed pool. About 10% of the remaining connected groups go to
``split="val"``.

    PYTHONPATH=. python build_split_v1.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
from pathlib import Path

from ctcbench_real.env import (_L32, DEFAULT_TRAIN_DIR, ROSTER, TRAIN_DIR_ENV, _flat_ints, _gold_of, _half, _hf_rung,
                               _train_rows, _unit_keys)
from ctcbench_real._vendor.ctc.format import registry

OUT = Path(__file__).parent / "ctcbench_real" / "split_v1.json"
VAL_MOD = 10   # 1 in VAL_MOD groups goes to val
# tasks built from the same seed pool as the key: their test rows also block the key's train rows
CROSS = {"nq": ["qdmatch_nq"], "qdmatch_nq": ["nq"], "hotpotqa": ["qdmatch_hpqa"], "qdmatch_hpqa": ["hotpotqa"],
         "niah": ["contradiction_iid"], "contradiction_iid": ["niah"], "xabsence": ["contradiction_iid", "niah"],
         "outlier": ["outlier_fixedM"], "outlier_fixedM": ["outlier"],
         "absence_gutenberg": ["reorder"], "reorder": ["absence_gutenberg"]}
RETIRED = {"qdmatch_nq": "train pairs are a reshuffle of the published eval pairs"}
# the tasks that have a v1 train / val split
TASKS = ["absence_gutenberg", "contradiction_iid", "grouping", "hotpotqa", "niah", "nq", "outlier_fixedM", "outlier",
         "qdmatch_hpqa", "qdmatch_nq", "reorder", "strmatch", "textgroups", "xabsence"]

H = lambda s: hashlib.sha1(s.encode()).hexdigest()[:16]  # noqa: E731


def gold_idx(ex: dict, spec) -> list[int]:
    docs = ex.get("documents") or []
    if spec.name == "reorder":
        return list(range(len(docs)))
    b = spec.gold_index_base
    if spec.name == "qdmatch":
        return sorted({i - b for p in (ex.get("gold_pairs") or []) for i in p if 0 <= i - b < len(docs)})
    return [i - b for i in _flat_ints(_gold_of(ex, spec)) if 0 <= i - b < len(docs)]


def source(ex: dict) -> str | None:
    m = ex.get("meta") or ex.get("_meta") or {}
    if isinstance(m, dict) and m.get("book"):
        return "book:" + str(m["book"])
    return ("src:" + str(ex["source_id"])) if ex.get("source_id") else None


def test_index(task: str) -> tuple[set, set, set]:
    """(gold doc hashes, context doc hashes, source ids) of the task's test rows."""
    row, spec = ROSTER[task], registry.get(ROSTER[task].spec)
    g, c, s = set(), set(), set()
    for r in row.rungs:
        if r not in _L32:
            continue
        for ex in _hf_rung(task, r):
            if row.train is None and _half(ex, spec) == 0:
                continue
            ds = [d["text"] for d in ex["documents"]]
            c |= {H(x) for x in ds}
            g |= {H(ds[i]) for i in gold_idx(ex, spec)}
            if source(ex):
                s.add(source(ex))
    return g, c, s


def uid(ex: dict, spec) -> str:
    return min(k for k in _unit_keys(ex, spec) if k.startswith("g:"))


def main() -> None:
    idx = {}
    out = {"version": "v1", "val_mod": VAL_MOD, "cross": CROSS, "retired": RETIRED, "tasks": {}}
    for t in TASKS:
        spec = registry.get(ROSTER[t].spec)
        for u in [t] + CROSS.get(t, []):
            if u not in idx:
                idx[u] = test_index(u)
        own_g, own_c, own_s = idx[t]
        x_g = set().union(*(idx[u][0] for u in CROSS.get(t, [])), set())
        x_s = set().union(*(idx[u][2] for u in CROSS.get(t, [])), set())
        rows = _train_rows(t, os.environ.get(TRAIN_DIR_ENV) or DEFAULT_TRAIN_DIR, ("r8k", "r16k", "r32k"))
        drop, keep = set(), []
        for ex in rows:
            ds = [H(d["text"]) for d in ex["documents"]]
            gs = {ds[i] for i in gold_idx(ex, spec)}
            s = source(ex)
            if t in RETIRED or gs & (own_g | own_c | x_g) or (s and s in (own_s | x_s)):
                drop.add(uid(ex, spec))
            else:
                keep.append((uid(ex, spec), gs | ({s} if s else set())))
        parent: dict[str, str] = {}

        def find(a: str) -> str:
            while parent.setdefault(a, a) != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for u_, ks in keep:
            for k in ks:
                ra, rb = find("u" + u_), find(k)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        comp = {u_: find("u" + u_) for u_, _ in keep}
        sizes = collections.Counter(comp.values())
        if keep and max(sizes.values()) > 0.2 * len(keep):
            comp = {u_: "u" + u_ for u_ in comp}
        val = sorted(u_ for u_, c in comp.items() if int(hashlib.sha1(c.encode()).hexdigest()[:8], 16) % VAL_MOD == 0)
        n_val = sum(1 for u_, _ in keep if u_ in set(val))
        out["tasks"][t] = {"n_legacy": len(rows), "n_drop": len(rows) - len(keep), "n_val": n_val,
                           "n_train": len(keep) - n_val, "giant_component": len(comp) != len(sizes) and
                           comp == {u_: "u" + u_ for u_ in comp}, "drop": sorted(drop), "val": val}
        print(t, {k: v for k, v in out["tasks"][t].items() if k.startswith("n_")}, flush=True)
    OUT.write_text(json.dumps(out, indent=0, sort_keys=True) + "\n")
    print("wrote", OUT, file=sys.stderr)


if __name__ == "__main__":
    main()
