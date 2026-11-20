"""Mean reward of the K nearest training rollouts vs evaluation reward (Fig. jaccard).

Neighbours are found by Weighted Jaccard over whole completions (inputs for the Transformer) among training rollouts
with train.step <= eval.step, without looking at rewards; ties go to the earlier step.

    python neighbor_rewards.py [--outputs $PRIME_RL_DIR/outputs] [--k 3] [--workers 4]   # -> results/neighbor_rewards/
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simlib import runs as R  # noqa: E402
from simlib.trajectories import _join_completion  # noqa: E402
from simlib.weighted_neighbors import WeightedNeighborIndex, token_counts  # noqa: E402

DISTANCE = HERE / "results" / "distance"
REGEN_CACHE = HERE / "cache" / "regen_baseline_eval"
OUT = HERE / "results" / "neighbor_rewards"


def read_records(path):
    with path.open("rb") as source:
        for row, line in enumerate(source):
            yield row, json.loads(line)


def metadata(record, run, split, step, row):
    reward = record.get("reward")
    # nDCG can exceed 1 by roundoff; keep it
    if not isinstance(reward, (int, float)) or not np.isfinite(reward) or not -1e-12 <= reward <= 1 + 1e-12:
        raise ValueError(f"Missing or invalid reward: {run}/{split}@{step}/{row}")
    info = record.get("info") or {}
    if isinstance(info, str):
        info = json.loads(info)
    return {"uid": f"{run}/{split}@{step}/{row}", "step": step, "row": row,
            "example_id": record.get("example_id"),
            "problem_id": str(info.get("id", record.get("example_id"))),
            "reward": float(reward), "error": bool(record.get("error")),
            "truncated": bool(record.get("is_truncated"))}


def context(record):
    info = record.get("info") or {}
    if isinstance(info, str):
        info = json.loads(info)
    value = info["context"]
    text = "\n\n".join(value) if isinstance(value, list) else value
    assert isinstance(text, str) and text
    return text


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def train_files(rollouts: Path, eval_step: int) -> list[Path]:
    return sorted((p for p in rollouts.glob("step_*/train_rollouts.jsonl") if int(p.parent.name[5:]) <= eval_step),
                  key=lambda p: int(p.parent.name[5:]))


def agent_cell(slug, harness, outputs, distance, k):
    method = R.method_for(slug, harness)
    cache = json.loads((distance / slug / f"{method}.json").read_text())
    run = cache["meta"]["run_id"]
    selected = {r["uid"]: r for r in cache["per_eval"]}
    steps = {int(uid.split("/eval@")[1].split("/")[0]) for uid in selected}
    assert len(steps) == 1, "expected one evaluation checkpoint"
    eval_step = steps.pop()
    rollouts = R.run_dir(outputs, slug, harness) / "run_default" / "rollouts"
    paths = train_files(rollouts, eval_step)
    train, index = [], WeightedNeighborIndex()
    print(f"LOAD {slug} {harness}: {len(paths)} train files", flush=True)
    for path in paths:
        step = int(path.parent.name[5:])
        for row, record in read_records(path):
            counts = token_counts(_join_completion(record.get("completion") or []))
            train.append(dict(metadata(record, run, "train", step, row), n_tokens=index.add(counts)))
    index.finalize()
    eval_records = []
    for row, record in read_records(rollouts / f"step_{eval_step}" / "eval_rollouts.jsonl"):
        meta = metadata(record, run, "eval", eval_step, row)
        if meta["uid"] not in selected:
            continue
        assert abs(meta["reward"] - selected[meta["uid"]]["reward"]) < 1e-10
        counts = token_counts(_join_completion(record.get("completion") or []))
        eval_records.append((dict(meta, n_tokens=sum(counts.values())), counts))
    assert {r["uid"] for r, _ in eval_records} == set(selected)
    cases = []
    for meta, counts in eval_records:
        result = index.query(counts, k)
        neighbors = [dict(train[match["index"]], rank=rank, similarity=match["similarity"])
                     for rank, match in enumerate(result["neighbors"], 1)]
        cases.append(dict(meta, neighbors=neighbors,
                          neighbor_reward=float(np.mean([r["reward"] for r in neighbors])),
                          mean_similarity=float(np.mean([r["similarity"] for r in neighbors])),
                          boundary_tie=result["boundary_tie"]))
    rewards = np.array([r["reward"] for r in train])
    summary = dict(panel=slug, method=harness, run_id=run, eval_step=eval_step, n_train=len(train),
                   n_train_zero=int((rewards == 0).sum()), all_train_reward=float(rewards.mean()),
                   n_eval=len(cases), neighbor_reward=float(np.mean([c["neighbor_reward"] for c in cases])),
                   eval_reward=float(np.mean([c["reward"] for c in cases])),
                   mean_similarity=float(np.mean([c["mean_similarity"] for c in cases])),
                   boundary_ties=sum(c["boundary_tie"] for c in cases),
                   representation="completion_transcript", eval_pairing="per_example")
    print(f"DONE {slug} {harness}: {len(train)} train, {len(cases)} eval", flush=True)
    return summary, cases


def transformer_cell(slug, outputs, distance, regen_cache, k):
    rollouts = R.run_dir(outputs, slug, "transformer") / "run_default" / "rollouts"
    run = R.entry(outputs, slug, "transformer")["run_id"]
    if slug in R.OFFLINE_TRANSFORMER:
        # 10-step grid only (step 149 is not a curve point)
        curve = {s: v for s, v in R.offline_eval_curve(outputs, slug).items() if s % 10 == 0}
        if not curve:
            raise SystemExit(f"{slug}: no offline long-split evaluation of the Transformer run (eval_transformer.sh)")
        step = max(sorted(curve), key=curve.get)
        aggregate_reward, pairing = float(curve[step]), "aggregate_only"
    else:
        cached = json.loads((distance / slug / "transformer.json").read_text())
        step = int(cached["meta"]["eval_step_causal"])
        aggregate_reward, pairing = float(np.mean([r["reward"] for r in cached["per_eval"]])), "per_example"
    texts, train = {}, []
    for path in train_files(rollouts, step):
        train_step = int(path.parent.name[5:])
        for row, record in read_records(path):
            text = context(record)
            key = digest(text)
            if key not in texts:
                texts[key] = (len(texts), text)
            train.append(dict(metadata(record, run, "train", train_step, row), context_index=texts[key][0]))
    records = []
    if pairing == "per_example":
        for row, record in read_records(rollouts / f"step_{step}" / "eval_rollouts.jsonl"):
            meta = metadata(record, run, "eval", step, row)
            records.append((dict(uid=meta["uid"], reward=meta["reward"], example_id=meta["example_id"]),
                            context(record)))
        assert abs(np.mean([m["reward"] for m, _ in records]) - aggregate_reward) < 1e-12
    elif slug == "oolong-pairs":
        # inputs only, not rewards
        last = max(int(p.parent.name[5:]) for p in rollouts.glob("step_*/eval_rollouts.jsonl"))
        for row, record in read_records(rollouts / f"step_{last}" / "eval_rollouts.jsonl"):
            records.append((dict(uid=f"{run}/eval_context@{step}/{row}", reward=None,
                                 example_id=record.get("example_id")), context(record)))
    else:
        payload = json.loads((regen_cache / f"{slug}.json").read_text())
        for r in payload["prompts"]:
            records.append((dict(uid=f"{run}/eval_context@{step}/{r['idx']}", reward=None,
                                 example_id=r["example_id"]), r["context"]))
    print(f"{slug} transformer: {len(train)} train rows, {len(texts)} unique contexts, {len(records)} eval contexts",
          flush=True)
    bank = [value[1] for value in texts.values()]
    index = WeightedNeighborIndex()
    for text in bank:
        index.add(token_counts(text))
    index.finalize()
    indices = np.arange(len(train))
    context_ids = np.array([t["context_index"] for t in train])
    rewards = np.array([t["reward"] for t in train])
    assert len(indices) >= k
    cases = []
    for meta, text in records:
        scores = np.empty(len(bank), dtype=float)
        for match in index.query(token_counts(text), k=len(bank))["neighbors"]:
            scores[match["index"]] = match["similarity"]
        similarities = scores[context_ids]
        order = np.lexsort((indices, -similarities))
        chosen = indices[order[:k]]
        threshold = similarities[order[k - 1]]
        needed = k - int((similarities > threshold).sum())
        neighbors = [dict(train[i], rank=rank, similarity=float(similarities[i])) for rank, i in enumerate(chosen, 1)]
        cases.append(dict(meta, neighbors=neighbors, neighbor_reward=float(rewards[chosen].mean()),
                          mean_similarity=float(np.mean([n["similarity"] for n in neighbors])),
                          boundary_tie=bool((similarities == threshold).sum() > needed)))
    summary = dict(panel=slug, method="transformer", run_id=run, eval_step=step, n_train=len(indices),
                   n_train_zero=int((rewards == 0).sum()), all_train_reward=float(rewards.mean()),
                   n_eval=len(cases), neighbor_reward=float(np.mean([c["neighbor_reward"] for c in cases])),
                   eval_reward=aggregate_reward,
                   mean_similarity=float(np.mean([c["mean_similarity"] for c in cases])),
                   boundary_ties=sum(c["boundary_tie"] for c in cases),
                   representation="input_context", eval_pairing=pairing)
    return summary, cases


def run_cell(job):
    slug, harness, args = job
    if harness == "transformer":
        return transformer_cell(slug, args.outputs, args.distance, args.regen_cache, args.k)
    return agent_cell(slug, harness, args.outputs, args.distance, args.k)


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=R.DEFAULT_OUTPUTS, help="prime-rl outputs directory")
    ap.add_argument("--distance", type=Path, default=DISTANCE, help="run_distance.py output directory")
    ap.add_argument("--regen-cache", type=Path, default=REGEN_CACHE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--panels", nargs="+", choices=list(R.PANEL), default=list(R.PANEL))
    ap.add_argument("--harnesses", nargs="+", choices=list(R.HARNESSES), default=list(R.HARNESSES))
    ap.add_argument("--workers", type=int, default=1, help="cells computed in parallel")
    args = ap.parse_args()
    jobs = [(slug, harness, args) for slug in args.panels for harness in args.harnesses]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
            cells = list(pool.map(run_cell, jobs))
    else:
        cells = [run_cell(job) for job in jobs]
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = [dict(metric="weighted_jaccard", k=args.k, **summary) for summary, _ in cells]
    neighbors = [dict(metric="weighted_jaccard", panel=summary["panel"], method=summary["method"],
                      run_id=summary["run_id"], eval_uid=case["uid"], eval_reward=case["reward"],
                      **{key: n[key] for key in ("uid", "rank", "similarity", "reward", "step", "row",
                                                 "example_id", "problem_id")})
                 for summary, cases in cells for case in cases for n in case["neighbors"]]
    write_csv(args.out / f"paired_summary_k{args.k}.csv", summaries)
    write_csv(args.out / f"neighbors_k{args.k}.csv", neighbors)
    for s in summaries:
        print(f"{s['panel']:26s} {s['method']:12s} neighbor={s['neighbor_reward']:.3f} eval={s['eval_reward']:.3f} "
              f"sim={s['mean_similarity']:.3f} n_eval={s['n_eval']} n_train={s['n_train']}")


if __name__ == "__main__":
    main()
