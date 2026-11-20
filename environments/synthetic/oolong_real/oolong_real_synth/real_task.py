"""The real OOLONG-Real D&D task as an RLMTrainEnv with a real sub-LM (used by the eval driver and RL validation).

The ~9 GB upstream jsonl is streamed and only the selected rows' windows are kept. Reward is the paper metric
(``scoring.score_paper``).
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset

import rlm_train

from .gen import root_prompt_for
from .scoring import score_paper

_MARK = re.compile(r"^\[START OF EPISODE\]$", re.M)


def stream_rows(path: str | Path, keep: Any = None, limit_windows: int | None = None) -> Iterable[dict]:
    """Yield rows (with their context window text) whose metadata passes ``keep(row_without_text)``."""
    windows = 0
    last_cw = None
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            r = json.loads(line)
            if r["context_window_id"] != last_cw:
                last_cw = r["context_window_id"]
                windows += 1
                if limit_windows is not None and windows > limit_windows:
                    return
            meta = {k: v for k, v in r.items() if k != "context_window_text"}
            if keep is None or keep(meta):
                yield r


def select_rows(path: str | Path, *, ids: list[str] | None = None, n: int = 6, max_episodes: int | None = 1,
                question_types: list[str] | None = None, seed: int = 0, limit_windows: int | None = None,
                per_window: int | None = None) -> list[dict]:
    """Explicit ``ids``, else a seeded sample of ``n`` rows from windows with <= ``max_episodes`` episodes."""
    if ids:
        want = set(ids)
        return list(stream_rows(path, lambda m: m["id"] in want, limit_windows))

    def keep(m: dict) -> bool:
        if max_episodes is not None and len(m["episodes"]) > max_episodes:
            return False
        return not question_types or m["question_type"] in question_types

    rows = list(stream_rows(path, keep, limit_windows))
    rng = random.Random(seed)
    if per_window:
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r["context_window_id"], []).append(r)
        rows = [x for v in by.values() for x in rng.sample(v, min(per_window, len(v)))]
    rng.shuffle(rows)
    return rows[:n]


def _n_episodes(text: str) -> int:
    return max(1, len(_MARK.findall(text)))


def build_dataset(rows: list[dict]) -> Dataset:
    out = []
    for i, r in enumerate(rows):
        n_eps = _n_episodes(r["context_window_text"])
        out.append({
            "example_id": i,
            "prompt": [{"role": "user", "content": r["question"]}],
            "answer": str(r["answer"]),
            "info": {"context": r["context_window_text"], "root_prompt": root_prompt_for(r["question"], n_eps),
                     "id": r["id"], "context_window_id": r["context_window_id"], "question": r["question"],
                     "question_type": r["question_type"], "episodes": r["episodes"], "campaign": r["campaign"],
                     "gold": str(r["answer"])},
        })
    return Dataset.from_list(out)


async def _correctness(info: Any = None, state: Any = None, **_kw: Any) -> float:
    meta = json.loads(info) if isinstance(info, str) else (info or {})
    final = (state or {}).get("rlm_final_answer")
    if final is None:
        return 0.0
    return float(score_paper(str(final), str(meta.get("gold", ""))))


def load_environment(path: str | None = None, *, rows: list[dict] | None = None, ids: list[str] | None = None,
                     num_examples: int = 6, max_episodes: int | None = 1, question_types: list[str] | None = None,
                     seed: int = 0, limit_windows: int | None = None, per_window: int | None = None,
                     max_iterations: int = 20, max_sub_calls: int = 600, sub_max_tokens: int = 4096,
                     sub_temperature: float | None = None,
                     **kwargs: Any) -> rlm_train.RLMTrainEnv:
    if rows is None:
        if path is None:
            raise ValueError("pass `path` (the upstream dnd/<split>.jsonl) or preselected `rows`")
        rows = select_rows(path, ids=ids, n=num_examples, max_episodes=max_episodes, question_types=question_types,
                           seed=seed, limit_windows=limit_windows, per_window=per_window)
    sub_args = {"max_tokens": sub_max_tokens}
    if sub_temperature is not None:
        if isinstance(sub_temperature, bool) or not 0 <= sub_temperature <= 2:
            raise ValueError("sub_temperature must be between zero and two")
        sub_args["temperature"] = sub_temperature
    rubric = rlm_train.RLMTrainRubric(correctness=_correctness, weight=1.0, min_iterations=1)
    kwargs.setdefault("code_block_feedback", True)  # Section 4 setting: feedback on turns without a code block
    return rlm_train.RLMTrainEnv(dataset=build_dataset(rows), max_iterations=max_iterations, rubric=rubric,
                                 sub_sampling_args=sub_args, max_sub_calls=max_sub_calls,
                                 **kwargs)


__all__ = ["build_dataset", "load_environment", "select_rows", "stream_rows"]
