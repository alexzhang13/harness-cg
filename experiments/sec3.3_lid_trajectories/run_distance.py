"""Similarity of each evaluation trajectory to its nearest training trajectories (Fig. similarity).

Each eval trajectory at the selected checkpoint is compared with every positive-reward training trajectory with
train.step <= eval.step, keeping the max per metric. The Transformer compares prompts instead of completions.
The published Pi OOLONG-Pairs cell used --cand-k 32 --weighted-jaccard exact, and the OBLIQ twitter -> wildchat RLM
cell --max-eval-step 150.

    python run_distance.py --all [--outputs $PRIME_RL_DIR/outputs]     # -> results/distance/<panel>/<method>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from simlib import distance as D  # noqa: E402
from simlib import metrics as M  # noqa: E402
from simlib import runs as R  # noqa: E402
from simlib import trajectories as TL  # noqa: E402

RESULTS = HERE / "results" / "distance"
REGEN_CACHE = HERE / "cache" / "regen_baseline_eval"
# The five metrics of Fig. similarity.
METRICS = ["word3_jaccard", "word3_containment", "token_levenshtein", "weighted_jaccard", "len_ratio"]


def run_series(slug: str, harness: str, entry: dict, metrics: list[str], cand_k: int,
               regen_cache: Path, weighted_jaccard: str = "shortlist") -> dict | None:
    run_id = entry["run_id"]
    is_baseline = harness == "transformer"
    eval_source = "inline"
    if is_baseline:
        # offline evals keep no prompts: use the regenerated ones if present
        cache = regen_cache / f"{slug}.json"
        if cache.exists():
            eval_trajs = D.build_eval_prompt_pool_from_cache(cache, run_id)
            eval_source = "regen-long"
        else:
            eval_trajs = D.build_eval_prompt_pool(entry)
    else:
        eval_trajs = TL.build_eval_pool(entry)
    if not eval_trajs:
        print(f"    [skip] {harness}={run_id}: empty eval pool")
        return None

    # train.step <= eval.step; regenerated prompts (step -1) use the selected step
    eval_step = None
    train_steps = entry["train"]["available_steps"]
    n_train_steps_all = len(train_steps)
    steps = [t.step for t in eval_trajs if t.step >= 0]
    eval_step = max(steps) if steps else entry["eval"].get("best_step")
    if eval_step is not None:
        if any(t.step < 0 for t in eval_trajs):
            for t in eval_trajs:
                if t.step < 0:
                    t.step = int(eval_step)
        train_steps = [s for s in train_steps if s <= int(eval_step)]

    if is_baseline:
        train_trajs = D.build_train_prompt_pool(entry, steps=train_steps)
    else:
        train_trajs = TL.build_train_pool(entry, steps=train_steps)
    if not train_trajs:
        print(f"    [skip] {harness}={run_id}: empty train pool"
              + (f" after causal filter (eval_step={eval_step})" if eval_step is not None else ""))
        return None

    sources = M.candidate_sources_for(metrics)
    shortlist = ("weighted_jaccard",) if weighted_jaccard == "shortlist" else ()
    res = D.measure(
        eval_trajs, train_trajs, metrics, cand_k=cand_k,
        require_train_before_eval=True,
        candidate_sources=sources, shortlist=shortlist,
    )
    eval_file_step = None
    if eval_trajs and eval_source != "regen-long":
        eval_file_step = eval_trajs[0].step
    res["meta"] = {
        "series": harness, "run_id": run_id, "mode": "prompt" if is_baseline else "rollout",
        "task": entry.get("task"), "method": entry.get("method"),
        "best_step": entry["eval"].get("best_step"),
        "best_score": entry["eval"].get("best_score"),
        "eval_file_step": eval_file_step,
        "eval_source": eval_source,
        "require_train_before_eval": True,
        "eval_step_causal": eval_step,
        "n_train_steps_all": n_train_steps_all,
        "n_train_steps_used": len(train_steps),
        "n_train_causal": len(train_trajs),
        "train_reward": "positive",
        "candidate_sources": list(sources),
        "cand_k": cand_k,
        "weighted_jaccard": weighted_jaccard,
    }
    sm = res["summary"]
    print(f"    {harness:12s} {run_id:48s} n_eval={res['n_eval']:3d} n_train={res['n_train']:4d}"
          + (f" (<=step {eval_step}, {len(train_steps)}/{n_train_steps_all} steps)"
             if eval_step is not None else f" (all {n_train_steps_all} train steps)")
          + "  "
          + "  ".join(f"{m}:sim={sm[m]['mean_sim']:.3f}" for m in metrics if sm.get(m)))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", action="append", default=[], choices=list(R.PANEL))
    ap.add_argument("--all", action="store_true", help="all eight panels")
    ap.add_argument("--harnesses", nargs="+", default=list(R.HARNESSES), choices=list(R.HARNESSES))
    ap.add_argument("--metrics", nargs="+", default=METRICS, help=f"subset of {M.all_metric_names()}")
    ap.add_argument("--cand-k", type=int, default=10, help="candidates per eval item for rerank metrics")
    ap.add_argument("--weighted-jaccard", choices=["shortlist", "exact"], default="shortlist",
                    help="Weighted Jaccard on the cand-k shortlist (published figure, all cells but one) or over "
                         "the whole training pool")
    ap.add_argument("--max-eval-step", type=int, help="select the checkpoint among evaluation steps <= this")
    ap.add_argument("--outputs", type=Path, default=R.DEFAULT_OUTPUTS, help="prime-rl outputs directory")
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--regen-cache", type=Path, default=REGEN_CACHE)
    args = ap.parse_args()
    panels = list(R.PANEL) if args.all else args.panel
    if not panels:
        ap.error("pass --panel <slug> (repeatable) or --all")
    print(f"metrics: {args.metrics}\noutputs: {args.outputs}\n")
    for slug in panels:
        print(f"[{slug}]")
        out_dir = args.results / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        for harness in args.harnesses:
            entry = R.entry(args.outputs, slug, harness, args.max_eval_step)
            if entry is None:
                print(f"    [skip] {harness}: no rollouts under {R.run_dir(args.outputs, slug, harness)}")
                continue
            res = run_series(slug, harness, entry, args.metrics, args.cand_k, args.regen_cache,
                             args.weighted_jaccard)
            if res is not None:
                (out_dir / f"{R.method_for(slug, harness)}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
