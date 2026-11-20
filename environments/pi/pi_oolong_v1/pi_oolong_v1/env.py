"""Pi baseline for OOLONG (`pi-oolong-v1`; Sections 3.1 and 3.2).

Rows and scoring from oolong_real; the context is context.txt.
"""

from __future__ import annotations

from typing import Any

import verifiers as vf

from rlm_train.pi import load_backing_env, make_pi_env

_backing = load_backing_env("oolong_real", "environments/real/oolong/oolong_real/env.py")

ANSWER_FORMAT = (
    "Write your final answer on its own line, prefixed with `Answer: ` — for "
    "example `Answer: entity`, `Answer: 42`, or `Answer: more common than`. "
    "Put only the answer after the colon, with no extra words."
)


def _score(output: str, gold: Any, meta: dict) -> float:
    datapoint = {
        "answer": meta.get("answer", gold),
        "answer_type": meta.get("answer_type", ""),
    }
    return _backing._synth_score(datapoint, output)


def load_environment(
    *,
    dataset_name: str = "trec_coarse",
    min_ctx: int = 1024,
    max_ctx: int = 4096,
    num_examples: int = -1,
    seed: int = 42,
    exclude_numeric: bool = False,
    max_turns: int = 30,
    **harness_kwargs: Any,
) -> vf.Environment:
    def build_rows() -> list[dict]:
        ds = _backing._build_dataset(
            dataset_name=dataset_name,
            min_ctx=min_ctx,
            max_ctx=max_ctx,
            num_examples=num_examples,
            seed=seed,
            exclude_numeric=exclude_numeric,
        )
        return list(ds)

    return make_pi_env(
        build_rows=build_rows,
        score_fn=_score,
        answer_format=ANSWER_FORMAT,
        reward_name="oolong_score",
        harness_kwargs={"max_turns": max_turns, **harness_kwargs},
    )
