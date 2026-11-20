"""Pi baseline for MRCR v2 (`pi-mrcr-v2-v1`; Section 3.1).

Rows and scoring from mrcr_v2_real; the transcript is context.txt.
"""

from __future__ import annotations

from typing import Any

import verifiers as vf

from rlm_train.pi import load_backing_env, make_pi_env

_backing = load_backing_env("mrcr_v2_real", "environments/real/mrcr_v2/mrcr_v2_real/env.py")

ANSWER_FORMAT = (
    "Your final answer MUST begin with the exact 12-character hash prefix given "
    "in the task instructions, immediately followed by the reproduced text — "
    "e.g. `a1b2c3d4e5f6<the reproduced response>`. Emit the hash+text last, with "
    "nothing after it."
)


def _score(output: str, gold: Any, meta: dict) -> float:
    return _backing._mrcr_score(output, str(gold))


def load_environment(
    *,
    needle_count: int = 2,
    context_range: str = "32k-64k",
    num_examples: int = -1,
    seed: int | None = 42,
    max_turns: int = 30,
    **harness_kwargs: Any,
) -> vf.Environment:
    def build_rows() -> list[dict]:
        return list(
            _backing._build_dataset(
                needle_count=needle_count,
                context_range=context_range,
                num_examples=num_examples,
                seed=seed,
            )
        )

    return make_pi_env(
        build_rows=build_rows,
        score_fn=_score,
        answer_format=ANSWER_FORMAT,
        reward_name="mrcr_ratio",
        harness_kwargs={"max_turns": max_turns, **harness_kwargs},
    )
