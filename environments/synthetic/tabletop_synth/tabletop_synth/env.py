"""``load_environment``: synthetic tabletop windows as an ``RLMTrainEnv`` whose sub-LM is the real model."""

from __future__ import annotations

import json
from typing import Any

from datasets import Dataset

import rlm_train

from ._borrow import score_synth
from .generate import generate


def build_dataset(num_examples: int, seed: int = 0, **gen_kw: Any) -> Dataset:
    rows = []
    for i in range(num_examples):
        g = generate(seed + i, **gen_kw)
        rows.append({
            "example_id": i,
            "prompt": [{"role": "user", "content": g["question"]}],
            "answer": g["answer"],
            "info": {"context": g["context"], "root_prompt": g["root_prompt"], "gold": g["answer"],
                     "question": g["question"], "qtype": g["qtype"], "family": g["meta"]["family"],
                     "seed": seed + i, "n_episodes": g["meta"]["n_episodes"],
                     "n_rolls": len(g["meta"]["rolls"]), "n_spells": len(g["meta"]["spells"])},
        })
    return Dataset.from_list(rows)


async def _correctness(info: Any = None, state: Any = None, **_kw: Any) -> float:
    meta = json.loads(info) if isinstance(info, str) else (info or {})
    final = (state or {}).get("rlm_final_answer")
    return 0.0 if final is None else float(score_synth(str(final), str(meta.get("gold", ""))))


def load_environment(num_examples: int = 200, seed: int = 0, *, n_episodes: int | None = None,
                     ep_lines: tuple[int, int] = (700, 1400), tid: str | None = None, numwords: float = 0.25,
                     surfaces: str = "v1", max_iterations: int = 20, max_sub_calls: int = 600,
                     sub_max_tokens: int = 4096,
                     sub_temperature: float | None = None, **kwargs: Any) -> rlm_train.RLMTrainEnv:
    ds = build_dataset(num_examples, seed, n_episodes=n_episodes, ep_lines=tuple(ep_lines), tid=tid,
                       numwords=numwords, surfaces=surfaces)
    sub_args: dict[str, Any] = {"max_tokens": sub_max_tokens}
    if sub_temperature is not None:
        sub_args["temperature"] = sub_temperature
    rubric = rlm_train.RLMTrainRubric(correctness=_correctness, weight=1.0, min_iterations=1)
    kwargs.setdefault("code_block_feedback", True)  # Section 4 setting: feedback on turns without a code block
    return rlm_train.RLMTrainEnv(dataset=ds, max_iterations=max_iterations, rubric=rubric,
                                 sub_sampling_args=sub_args, max_sub_calls=max_sub_calls, **kwargs)
