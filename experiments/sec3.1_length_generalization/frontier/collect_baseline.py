"""Mean reward per task into baseline.json.

    python collect_baseline.py [results_dir]     # default results/gpt-6-astra, else the published results from HF
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "gpt-6-astra"
OUT = HERE / "baseline.json"
EXP_LOGS = ("a1zhang/harness_compgen_exp_logs", "7e3e9b15c7bd97ae975a9c4f3a668c1b50f51777")

PANEL_KEY = {
    "mrcr": "MRCRv2",
    "graphwalks": "GraphWalks",
    "longbenchpro": "LongBenchPro",
    "oolong": "OOLONG",
    "oolong_pairs": "OOLONG-Pairs",
    "adaleval_bestanswer": "AdaLEval-BestAnswer",
}


def main() -> None:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    if len(sys.argv) == 1 and not any(RESULTS.glob("*.json")):
        from huggingface_hub import snapshot_download
        root = snapshot_download(EXP_LOGS[0], repo_type="dataset", revision=EXP_LOGS[1],
                                 allow_patterns="length-gen/frontier/gpt-6-astra/*.json")
        results = Path(root) / "length-gen/frontier/gpt-6-astra"
    out: dict[str, dict] = {}
    for task, panel in PANEL_KEY.items():
        path = results / f"{task}.json"
        if not path.exists():
            print(f"  (missing) {task}: no {path.name} yet")
            continue
        d = json.loads(path.read_text())
        s = d.get("summary", {})
        if not s.get("n"):
            print(f"  (empty) {task}: no scored rollouts")
            continue
        out[panel] = {
            "mean_reward": s["mean_reward"],
            "n": s["n"],
            "errors": s.get("errors", 0),
            "model": d["model"],
            "label": f"{d['model']} (no hint)",
            "task": task,
        }
        print(f"  {panel:16s} mean_reward={s['mean_reward']:.3f}  "
              f"n={s['n']} (+{s.get('errors', 0)}e)  [{d['model']}]")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}  ({len(out)}/{len(PANEL_KEY)} tasks)")


if __name__ == "__main__":
    main()
