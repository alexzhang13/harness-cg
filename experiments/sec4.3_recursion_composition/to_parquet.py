"""SFT samples (build_sft.py output) -> <out dir>/{train,validation}.parquet for prime-rl's SFT trainer.

    python to_parquet.py runs/sft_gated.jsonl <out dir>

Validation = about 1/16 of the chains. Thinking disabled.
"""
import hashlib, json, random, sys
from pathlib import Path
import pandas as pd

src, dest = sys.argv[1], Path(sys.argv[2])
rows = [json.loads(l) for l in open(src)]
chains = sorted({json.dumps(r["pool_ids"]) for r in rows})
random.Random(0).shuffle(chains)
val = set(chains[:max(1, len(chains) // 16)])
out = {"train": [], "validation": []}
for i, r in enumerate(rows):
    key = json.dumps(r["pool_ids"])
    out["validation" if key in val else "train"].append({
        "messages": r["messages"], "chat_template_kwargs": {"enable_thinking": False}, "env": "composition-stitch",
        "uid": f"stitch:{hashlib.sha1((key + str(i)).encode()).hexdigest()[:12]}#d{r['depth']}", "task_id": "stitch:" + key,
        "source_set": "composition_exps/stitch"})
dest.mkdir(parents=True, exist_ok=True)
for k, v in out.items():
    pd.DataFrame(v).to_parquet(dest / f"{k}.parquet"); print(k, len(v))
