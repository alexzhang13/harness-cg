"""Before / after row counts of the synth eval decontamination (ctcbench_synth.decontam), per family and split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ctcbench_synth import decontam as D  # noqa: E402
from ctcbench_synth.data import FAMILIES, examples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    rep = {}
    print("| family | split | before | after | dropped | eval_row | eval_gold | eval_article | eval_item | scrubbed (items removed) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for fam in FAMILIES:
        for split in ("train", "val"):
            raw = list(examples(fam, split, decontaminate=False))
            kept, c = D.filter_rows(fam, raw)
            rep[f"{fam}/{split}"] = {"before": len(raw), **c}
            print(f"| {fam} | {split} | {len(raw)} | {len(kept)} | {c['dropped']} | {c['eval_row']} | {c['eval_gold']} "
                  f"| {c['eval_article']} | {c['eval_item']} | {c['scrubbed']} ({c['items_removed']}) |", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
