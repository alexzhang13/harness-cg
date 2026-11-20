"""Pi baseline for Ada-LEval BestAnswer (`pi-ada-leval-v1`; Section 3.1).

Rows and label matching from ada_leval_real; question and candidates are context.txt.
"""

from __future__ import annotations

from typing import Any

import verifiers as vf

from rlm_train.pi import load_backing_env, make_pi_env

_backing = load_backing_env(
    "ada_leval_real", "environments/real/ada_leval/ada_leval_real/env.py"
)

_BA_FORMAT = (
    "Answer with ONLY the label of the single best candidate, e.g. `A17` (or "
    "`Answer: A17`). Do not include the candidate text."
)


def _score_ba(output: str, gold: Any, meta: dict) -> float:
    g = _backing._coerce_label(meta.get("answer", gold))
    p = _backing._coerce_label(output)
    return 1.0 if (p is not None and g is not None and p == g) else 0.0


def load_environment(
    *,
    task: str = "bestanswer",
    setting: str = "16k",
    num_examples: int = -1,
    data_dir: str | None = None,
    manifest_path: str | None = None,
    manifest_split: str | None = None,
    max_turns: int = 30,
    **harness_kwargs: Any,
) -> vf.Environment:
    if task != "bestanswer":
        raise ValueError(f"ada_leval task must be 'bestanswer', got {task!r}")
    data_dir = _backing._data_dir(data_dir)
    include_qids = None
    if manifest_path is not None or manifest_split is not None:
        include_qids = _backing._manifest_qids(manifest_path, manifest_split)

    def build_rows() -> list[dict]:
        return list(
            _backing._build_ba_dataset(
                setting=setting,
                num_examples=num_examples,
                data_dir=data_dir,
                include_qids=include_qids,
            )
        )

    return make_pi_env(
        build_rows=build_rows,
        score_fn=_score_ba,
        answer_format=_BA_FORMAT,
        reward_name="adaleval_bestanswer",
        harness_kwargs={"max_turns": max_turns, **harness_kwargs},
    )
