"""Pi baseline for GraphWalks (`pi-graphwalks-v1`; Section 3.1).

Rows and set F1 from graphwalks_real; graph and operation are context.txt.
"""

from __future__ import annotations

import json
from typing import Any

import verifiers as vf

from rlm_train.pi import load_backing_env, make_pi_env

_backing = load_backing_env(
    "graphwalks_real", "environments/real/graphwalks/graphwalks_real/env.py"
)

ANSWER_FORMAT = (
    "End your response with a single line of the exact form:\n"
    "    final answer: [\"<id1>\", \"<id2>\", ...]\n"
    "i.e. `final answer:` followed by a bracketed, comma-separated list of the "
    "node-id strings. For an empty result use `final answer: []`."
)


def _score(output: str, gold: Any, meta: dict) -> float:
    gold_nodes = json.loads(gold) if isinstance(gold, str) else gold
    return _backing._f1(_backing._to_nodes(output), _backing._to_nodes(gold_nodes))


def load_environment(
    *,
    split: str = "128k_and_shorter",
    num_examples: int = -1,
    seed: int = 42,
    max_turns: int = 30,
    **harness_kwargs: Any,
) -> vf.Environment:
    def build_rows() -> list[dict]:
        return list(
            _backing._build_dataset(
                split=split,
                num_examples=num_examples,
                seed=seed,
            )
        )

    return make_pi_env(
        build_rows=build_rows,
        score_fn=_score,
        answer_format=ANSWER_FORMAT,
        reward_name="graphwalks_f1",
        harness_kwargs={"max_turns": max_turns, **harness_kwargs},
    )
