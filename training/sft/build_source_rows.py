"""Whole-trajectory SFT source rows from a local snapshot of the synthetic trace dataset.

    python build_source_rows.py <snapshot dir> <out.jsonl> [--exclude-descriptor D ...] [--selected-fastest-only D ...]

<snapshot dir> is a download of a1zhang/synthetic-rlm-data-clean. Keeps clean rows; train_from_turn becomes
meta.reverify.live_from; LongReason sub-call rows are rendered with thinking on.
"""
import argparse
import collections
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("snapshot")
ap.add_argument("out")
ap.add_argument("--exclude-descriptor", action="append", default=[])
ap.add_argument("--selected-fastest-only", action="append", default=[],
                help="descriptor whose rows are kept only when selected_fastest == True")
a = ap.parse_args()

kept, skipped = collections.Counter(), collections.Counter()
with open(a.out, "w") as f:
    for path in sorted(glob.glob(os.path.join(a.snapshot, "data", "*.jsonl"))):
        cfg = os.path.basename(path)[:-6]
        for line in open(path):
            r = json.loads(line)
            d = r.get("descriptor") or ""
            if str(r.get("clean")) != "True":
                skipped[(cfg, d, "not_clean")] += 1
                continue
            if d in a.exclude_descriptor:
                skipped[(cfg, d, "excluded")] += 1
                continue
            if d in a.selected_fastest_only and str(r.get("selected_fastest")) != "True":
                skipped[(cfg, d, "not_selected_fastest")] += 1
                continue
            msgs = [{"role": m["role"], "content": m["content"], "reasoning_content": m.get("reasoning_content") or ""}
                    for m in r["messages"]]
            if not any(m["role"] == "assistant" for m in msgs):
                skipped[(cfg, d, "no_assistant")] += 1
                continue
            meta = {"env": cfg, "uid": f"{cfg}:{r.get('source_uid') or r.get('task_id')}", "task": d}
            row = {"messages": msgs, "meta": meta}
            if r.get("row_kind") == "subcall":
                row["chat_template_kwargs"] = {"enable_thinking": True}
            else:
                tft = int(r.get("train_from_turn", -1) or -1)
                if tft > 1:
                    meta["reverify"] = {"live_from": tft}
            f.write(json.dumps(row) + "\n")
            kept[(cfg, d)] += 1

summary = {"kept_total": sum(kept.values()), "skipped_total": sum(skipped.values()),
           "kept": {f"{c}/{d}": n for (c, d), n in sorted(kept.items())},
           "skipped": {f"{c}/{d}/{w}": n for (c, d, w), n in sorted(skipped.items())}}
json.dump(summary, open(a.out + ".summary.json", "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("kept_total", "skipped_total")}))
