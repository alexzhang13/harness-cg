"""Rebuild the long-split eval prompts of the offline-evaluated Transformer runs, which only store scores.

    python regen_baseline_eval.py --task oolong      # -> cache/regen_baseline_eval/<task>.json (prime-rl venv)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in ("environments/singleturn", "environments/real/oolong", "environments/real/graphwalks",
          "environments/real/longbenchpro", "environments/real/oolong_pairs"):
    sys.path.insert(0, str(REPO / p))

from singleturn_baseline import env as STE  # noqa: E402

CACHE = HERE / "cache" / "regen_baseline_eval"

# task -> (slug, loader, long-split args), as in eval_transformer.sh
TASKS = {
    "oolong": ("oolong", STE._load_oolong,
               dict(num_examples=50, seed=42, dataset_name="trec_coarse",
                    min_ctx=262144, max_ctx=262144)),
    "graphwalks": ("graphwalks", STE._load_graphwalks,
                   dict(num_examples=50, seed=42, split="256k_to_1mil")),
    "longbenchpro": ("longbenchpro", STE._load_longbenchpro,
                     dict(num_examples=115, seed=42, token_length="256k", language="English")),
}


def user_text(prompt_msgs: list[dict]) -> str:
    parts = [f"<user>\n{m.get('content') or ''}" for m in prompt_msgs if m.get("role") == "user"]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--out", type=Path, default=CACHE)
    args = ap.parse_args()
    slug, loader, kw = TASKS[args.task]
    print(f"[regen] {args.task} slug={slug} args={kw}", flush=True)
    data, _rubric = loader(system_prompt=STE._DEFAULT_SYSTEM, **kw)
    out = []
    for i, row in enumerate(data):
        info = row.get("info")
        info = json.loads(info) if isinstance(info, str) else (info or {})
        txt = user_text(row["prompt"])
        context = info.get("context", "")
        context = "\n\n".join(context) if isinstance(context, list) else context
        out.append({"idx": i, "example_id": info.get("id", i), "n_chars": len(txt), "text": txt,
                    "context": context})
    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out / f"{slug}.json"
    p.write_text(json.dumps({"task": args.task, "slug": slug, "args": kw, "prompts": out}))
    lens = sorted(r["n_chars"] for r in out)
    print(f"[regen] wrote {p}  n={len(out)} chars min={lens[0]} med={lens[len(lens)//2]} max={lens[-1]} "
          f"({p.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
