"""Pi baseline for OOLONG-Pairs (`pi-oolong-pairs-v1`; Section 3.1).

Rows and pair-set F1 from oolong_pairs_real; the context is context.txt.
"""

from __future__ import annotations

import json
import random
from typing import Any

import verifiers as vf

from rlm_train.pi import load_backing_env, make_pi_env

_backing = load_backing_env(
    "oolong_pairs_real", "environments/real/oolong_pairs/oolong_pairs_real/env.py"
)

ANSWER_FORMAT = (
    "List every matching pair of user IDs, one per line, each in the form "
    "`(id_lo, id_hi)` with the smaller ID first — e.g.\n    (10352, 12455)\n"
    "    (10352, 12646)\nList no duplicate pairs. If there are none, write `none`."
)


def _score(output: str, gold: Any, meta: dict) -> float:
    gold_list = json.loads(gold) if isinstance(gold, str) else (gold or [])
    return _backing._score_f1(output, gold_list)


def load_environment(
    *,
    context_length: int = 32768,
    num_examples: int = 20,
    shuffle: bool = True,
    seed: int | None = 42,
    max_turns: int = 30,
    **harness_kwargs: Any,
) -> vf.Environment:
    if context_length not in _backing.SUPPORTED_CONTEXT_LENGTHS:
        raise ValueError(
            f"context_length {context_length} not in {_backing.SUPPORTED_CONTEXT_LENGTHS}"
        )

    def build_rows() -> list[dict]:
        questions = list(_backing._load_pairs(context_length))
        if shuffle and seed is not None:
            random.Random(seed).shuffle(questions)
        if num_examples and num_examples > 0:
            questions = questions[:num_examples]
        unlabeled, _labeled = _backing._load_trec_coarse_context(context_length)
        rows: list[dict] = []
        for i, q in enumerate(questions):
            raw_question = q["question"]
            gold = q.get("answer", [])
            rows.append(
                {
                    "example_id": i,
                    "prompt": [{"role": "user", "content": raw_question}],
                    "answer": json.dumps(list(gold)),
                    "info": json.dumps(
                        {
                            "context": unlabeled,
                            "raw_question": raw_question,
                            "context_len": context_length,
                            "dataset": "oolong_pairs_trec_coarse_real",
                            "root_prompt": raw_question,
                        }
                    ),
                }
            )
        return rows

    return make_pi_env(
        build_rows=build_rows,
        score_fn=_score,
        answer_format=ANSWER_FORMAT,
        reward_name="oolong_pairs_f1",
        harness_kwargs={"max_turns": max_turns, **harness_kwargs},
    )
