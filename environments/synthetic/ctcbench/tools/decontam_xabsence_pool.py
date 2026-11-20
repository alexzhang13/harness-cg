"""Tighten the xabsence_paraphrase pool's eval exclusion from exact-sentence to word 13-gram (run after
build_xabsence_para_pool.py; needs ``eval_index.npz``). Rewrites the pool in place, keeping a ``.pre13gram`` backup."""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ctcbench_synth import decontam as D  # noqa: E402
from ctcbench_synth.xabsence_para import SIDECAR  # noqa: E402


def main() -> None:
    with gzip.open(SIDECAR, "rt", encoding="utf-8") as f:
        d = json.load(f)
    ev = D.index()["pubmed13"]
    keep, drop = [], []
    for a in d["abstracts"]:
        hit = any(g in ev for o, p in a["pairs"] for s in (o, p) for g in D.grams(s))
        (drop if hit else keep).append(a)
    print(f"abstracts {len(d['abstracts'])} -> {len(keep)} (dropped {len(drop)}: "
          f"{sum(len(a['pairs']) for a in drop)} pairs; train {sum(a['split'] == 'train' for a in drop)}, "
          f"val {sum(a['split'] == 'val' for a in drop)})")
    if not drop:
        return
    bak = SIDECAR.with_name(SIDECAR.name + ".pre13gram")
    if not bak.exists():
        shutil.copy2(SIDECAR, bak)
    d["abstracts"] = keep
    ex = dict(d.get("eval_exclusion") or {})
    ex["ngram13"] = {"ladders": ["xabsence", "xabsence_paraphrase", "contradiction_iid"], "rungs": "all published",
                     "abstracts_dropped": len(drop) + int(ex.get("ngram13", {}).get("abstracts_dropped", 0)),
                     "dropped_ids": sorted({a["id"] for a in drop} | set(ex.get("ngram13", {}).get("dropped_ids", [])))}
    d["eval_exclusion"] = ex
    with gzip.open(SIDECAR, "wt", encoding="utf-8") as f:
        json.dump(d, f)
    print("rewrote", SIDECAR, "(backup", bak, ")")


if __name__ == "__main__":
    main()
