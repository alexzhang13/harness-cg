"""Evaluation x training similarity matrices for a set of metrics (see metrics.py)."""

from __future__ import annotations

import numpy as np

from . import metrics as M
from .trajectories import Traj


def compute_matrices(
    eval_trajs: list[Traj],
    train_trajs: list[Traj],
    metric_names: list[str],
    cand_k: int = 10,
    candidate_sources: tuple[str, ...] | None = None,
    shortlist: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """{metric: (n_eval, n_train) similarity}. Metrics in `shortlist` are only scored on the candidate shortlist."""
    ec = M.Corpus([t.full_text for t in eval_trajs])
    tc = M.Corpus([t.full_text for t in train_trajs])

    dense = [n for n in metric_names if n in M.DENSE_METRICS and n not in shortlist]
    rerank = [n for n in metric_names if n in M.RERANK_METRICS and (n not in M.DENSE_METRICS or n in shortlist)]
    unknown = [n for n in metric_names if n not in M.DENSE_METRICS and n not in M.RERANK_METRICS]
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}. known: {M.all_metric_names()}")

    mats: dict[str, np.ndarray] = {}
    for name in dense:
        mats[name] = M.DENSE_METRICS[name](ec, tc)

    if rerank:
        # candidates: top cand_k by summed candidate-source similarity
        sources = candidate_sources if candidate_sources is not None else M.CANDIDATE_SOURCES
        src = np.zeros((len(eval_trajs), len(train_trajs)), dtype=np.float32)
        for s in sources:
            src += mats.get(s) if s in mats else M.DENSE_METRICS[s](ec, tc)
        tok_e, tok_t = ec.tokens(), tc.tokens()
        for name in rerank:
            fn = M.RERANK_METRICS[name]
            mat = np.zeros((len(eval_trajs), len(train_trajs)), dtype=np.float32)
            for i in range(len(eval_trajs)):
                for j in np.argsort(-src[i])[:cand_k]:
                    mat[i, j] = fn(tok_e[i], tok_t[j])
            mats[name] = mat
    return mats
