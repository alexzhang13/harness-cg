"""Nearest eligible training item per eval item, per metric (Fig. similarity).

Harnesses compare rollouts with positive-reward training rollouts; the Transformer compares prompts with the
prompts of training tasks that had a positive-reward sample.
"""

from __future__ import annotations

from pathlib import Path

from . import coverage as COV
from . import trajectories as TL
from .trajectories import Traj


def _prompt_text(rec: dict) -> str:
    """User messages of a rollout's prompt."""
    parts: list[str] = []
    for m in rec.get("prompt") or []:
        if m.get("role") == "user":
            parts.append(f"<user>\n{m.get('content') or ''}")
    return "\n".join(parts)


def build_eval_prompt_pool(entry: dict) -> list[Traj]:
    """One Traj per eval rollout at the selected step (prompt text)."""
    step = TL.resolve_eval_step(entry)
    path = Path(entry["rollout_root"]) / f"step_{step}" / "eval_rollouts.jsonl"
    out: list[Traj] = []
    for row, rec in enumerate(TL.iter_rollout_file(path)):
        comp = rec.get("completion") or []
        out.append(
            Traj(
                uid=f"{entry['run_id']}/eval@{step}/{row}",
                run_id=entry["run_id"], split="eval", step=step, row=row,
                example_id=int(rec.get("example_id", -1) or -1),
                reward=float(rec.get("reward", 0.0) or 0.0),
                num_turns=int(rec.get("num_turns", 0) or 0),
                rlm_iterations=int(rec.get("rlm_iterations", 0) or 0),
                n_msgs=len(comp), full_text=_prompt_text(rec),
            )
        )
    return out


def build_eval_prompt_pool_from_cache(cache_path, run_id: str) -> list[Traj]:
    """Eval prompts from the regen_baseline_eval.py cache."""
    import json
    from pathlib import Path

    payload = json.loads(Path(cache_path).read_text())
    out: list[Traj] = []
    for r in payload["prompts"]:
        out.append(
            Traj(
                uid=f"{run_id}/eval@regen/{r['idx']}", run_id=run_id, split="eval",
                step=-1, row=r["idx"], example_id=int(r.get("example_id", r["idx"]) or r["idx"]),
                reward=0.0, num_turns=1, rlm_iterations=0, n_msgs=1, full_text=r["text"],
            )
        )
    return out


def build_train_prompt_pool(entry: dict, steps: list[int] | None = None) -> list[Traj]:
    """One Traj per training task with a positive-reward sample."""
    root = Path(entry["rollout_root"])
    if steps is None:
        steps = entry["train"]["available_steps"]
    positive_ids: set[int] = set()
    first_seen: dict[int, tuple[int, int, dict]] = {}
    for s in steps:
        f = root / f"step_{s}" / "train_rollouts.jsonl"
        if not f.exists():
            continue
        for row, rec in enumerate(TL.iter_rollout_file(f)):
            eid = int(rec.get("example_id", -1) or -1)
            if eid < 0:
                continue
            if float(rec.get("reward", 0.0) or 0.0) > 0:
                positive_ids.add(eid)
            first_seen.setdefault(eid, (s, row, rec))
    out: list[Traj] = []
    for eid in sorted(positive_ids):
        if eid not in first_seen:
            continue
        s, row, rec = first_seen[eid]
        comp = rec.get("completion") or []
        out.append(
            Traj(
                uid=f"{entry['run_id']}/train@{s}/{row}",
                run_id=entry["run_id"], split="train", step=s, row=row,
                example_id=eid, reward=1.0,
                num_turns=int(rec.get("num_turns", 0) or 0),
                rlm_iterations=int(rec.get("rlm_iterations", 0) or 0),
                n_msgs=len(comp), full_text=_prompt_text(rec),
            )
        )
    return out


def measure(
    eval_trajs: list[Traj],
    train_trajs: list[Traj],
    metric_names: list[str],
    *,
    cand_k: int = 10,
    require_train_before_eval: bool = False,
    candidate_sources: tuple[str, ...] | None = None,
    shortlist: tuple[str, ...] = (),
) -> dict:
    """Nearest-train similarity per eval item and metric (optionally only train.step <= eval.step)."""
    import numpy as np

    mats = COV.compute_matrices(
        eval_trajs, train_trajs, metric_names, cand_k=cand_k,
        candidate_sources=candidate_sources, shortlist=shortlist,
    )
    if require_train_before_eval and train_trajs:
        train_steps = np.array([t.step for t in train_trajs], dtype=np.int64)
        for i, et in enumerate(eval_trajs):
            if et.step < 0:
                continue
            bad = train_steps > et.step
            if not bad.any():
                continue
            for mat in mats.values():
                mat[i, bad] = -np.inf

    per_eval = []
    for i, et in enumerate(eval_trajs):
        nn = {}
        for name, mat in mats.items():
            row = mat[i]
            if not row.size or not np.isfinite(row).any():
                nn[name] = {"sim": 0.0, "dist": 1.0, "match_uid": None}
                continue
            j = int(np.nanargmax(np.where(np.isfinite(row), row, -np.inf)))
            sim = float(row[j])
            nn[name] = {"sim": sim, "dist": 1.0 - sim,
                        "match_uid": train_trajs[j].uid if j >= 0 else None}
        per_eval.append({"uid": et.uid, "example_id": et.example_id,
                         "reward": et.reward, "num_turns": et.num_turns, "nn": nn})
    summary = {}
    for name, mat in mats.items():
        finite = np.where(np.isfinite(mat), mat, -np.inf)
        nn = finite.max(axis=1) if finite.size else np.array([])
        nn = np.where(np.isfinite(nn), nn, 0.0)
        dist = 1.0 - nn
        summary[name] = {
            "mean_dist": float(dist.mean()) if dist.size else None,
            "median_dist": float(np.median(dist)) if dist.size else None,
            "mean_sim": float(nn.mean()) if nn.size else None,
            "n": int(dist.size),
        }
    return {"n_eval": len(eval_trajs), "n_train": len(train_trajs),
            "per_eval": per_eval, "summary": summary,
            "require_train_before_eval": require_train_before_eval}
