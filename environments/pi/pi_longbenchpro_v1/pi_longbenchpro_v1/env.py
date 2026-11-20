"""Pi baseline for LongBench-Pro (`pi-longbenchpro-v1`; Section 3.1).

Rows and metrics from longbenchpro_real; scores the text after the last [Answer].
"""

from __future__ import annotations

from typing import Any

import verifiers as vf

from rlm_train.pi import load_backing_env, make_pi_env

_backing = load_backing_env(
    "longbenchpro_real", "environments/real/longbenchpro/longbenchpro_real/env.py"
)

ANSWER_FORMAT = (
    "Put a line containing exactly `[Answer]`, then your final answer on the "
    "line(s) that follow it. If the answer is a list of items, put each item on "
    "its own line. Only the text after the last `[Answer]` marker is scored."
)


def _score(output: str, gold: Any, meta: dict) -> float:
    return _backing._compute(meta.get("secondary_task", ""), meta.get("answers", []), output)


def load_environment(
    *,
    language: str = "all",
    token_length: str = "all",
    thinking: bool = False,
    num_examples: int = -1,
    seed: int | None = 42,
    max_turns: int = 30,
    **harness_kwargs: Any,
) -> vf.Environment:
    def build_rows() -> list[dict]:
        return list(
            _backing._build_dataset(
                language=language,
                token_length=token_length,
                thinking=thinking,
                num_examples=num_examples,
                seed=seed,
            )
        )

    return make_pi_env(
        build_rows=build_rows,
        score_fn=_score,
        answer_format=ANSWER_FORMAT,
        reward_name="longbenchpro_score",
        harness_kwargs={"max_turns": max_turns, **harness_kwargs},
    )
