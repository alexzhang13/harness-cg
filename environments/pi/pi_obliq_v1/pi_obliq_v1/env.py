"""Pi baseline for OBLIQ-Bench (`pi-obliq-v1`; Section 3.2).

Rows and metrics from obliq_real; the pool is context.txt. Reward = `metric` (nDCG@50), other metrics logged.
"""

from __future__ import annotations

from typing import Any

import verifiers as vf

from rlm_train.pi import load_backing_env, make_pi_env

_backing = load_backing_env("obliq_real", "environments/real/obliq/obliq_real/env.py")

ANSWER_FORMAT = (
    "Output your final ranking as a single Python list of bare doc_id strings, "
    "MOST relevant first — e.g. `[\"id_a\", \"id_b\", \"id_c\", ...]`. Use the bare "
    "id (the text after `doc_id:` in each candidate), rank ALL candidates, and "
    "put uncertain ones lower rather than dropping them (so borderline positives "
    "still count toward recall@50/@100)."
)


def _rels(meta: dict) -> dict:
    return {str(c): int(s) for c, s in (meta.get("relevant") or {}).items()}


def _bundle(output: str, meta: dict) -> dict:
    ranking = _backing._parse_ranking(output)
    return _backing._all_retrieval_metrics(ranking, _rels(meta))


def _metric_scorer(key: str):
    def _score(output: str, gold: Any, meta: dict) -> float:
        rels = _rels(meta)
        if not rels:
            return 0.0
        return float(_bundle(output, meta).get(key, 0.0))

    return _score


_AUX_KEYS = ("ndcg@50", "ndcg@10", "recall@10", "recall@50", "recall@100")
_AUX_NAMES = {
    "ndcg@50": "obliq_ndcg50",
    "ndcg@10": "obliq_ndcg10",
    "recall@10": "obliq_recall10",
    "recall@50": "obliq_recall50",
    "recall@100": "obliq_recall100",
}


def load_environment(
    *,
    task: str = "math",
    num_examples: int = 20,
    seed: int = 42,
    pool_size: int = 50,
    min_distractors: int = 10,
    metric: str = "ndcg@50",
    doc_max_chars: int = 4000,
    max_scan: int | None = 250_000,
    max_turns: int = 30,
    **harness_kwargs: Any,
) -> vf.Environment:
    def build_rows() -> list[dict]:
        return list(
            _backing._build_task_rows(
                task=task,
                num_examples=num_examples,
                seed=seed,
                pool_size=pool_size,
                min_distractors=min_distractors,
                metric=metric,
                doc_max_chars=doc_max_chars,
                max_scan=max_scan,
                start_id=0,
            )
        )

    aux = [(_AUX_NAMES[k], _metric_scorer(k)) for k in _AUX_KEYS if k != metric]

    return make_pi_env(
        build_rows=build_rows,
        score_fn=_metric_scorer(metric),
        answer_format=ANSWER_FORMAT,
        reward_name=_AUX_NAMES.get(metric, "obliq_score"),
        aux_metrics=aux,
        harness_kwargs={"max_turns": max_turns, **harness_kwargs},
    )
