"""Extract OOLONG-real D&D windows. Only used to build the real cast-name list that export_sft.py rejects.

    python prepare.py --episodes 3"""
import argparse
import json
import os
from pathlib import Path

DATA = Path(os.environ.get("OOLONG_REAL_DATA_DIR") or Path.home() / "oolong_real_data") / "dnd"
OUT = Path(os.environ.get("OOLONG_LMPROG_DATA") or Path(__file__).resolve().parents[2] / "outputs/synth-datagen/oolong_real/data")

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, nargs="+", default=[3])
a = ap.parse_args()
(OUT / "contexts").mkdir(parents=True, exist_ok=True)
for split in ("validation", "test"):
    outs = {n: open(OUT / f"{split}_{n}ep.jsonl", "w") for n in a.episodes}
    seen = set()
    with open(DATA / f"{split}.jsonl", "rb") as f:
        for line in f:
            r = json.loads(line)
            n = len(r["episodes"])
            if n not in outs:
                continue
            wid = r["context_window_id"]
            if wid not in seen:
                seen.add(wid)
                (OUT / "contexts" / f"{wid}.txt").write_text(r["context_window_text"])
            outs[n].write(json.dumps({k: v for k, v in r.items() if k != "context_window_text"}) + "\n")
    for fh in outs.values():
        fh.close()
    print(split, "windows", len(seen), flush=True)
