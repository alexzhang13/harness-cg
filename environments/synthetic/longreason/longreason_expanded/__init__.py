"""longreason-expanded: LongReason-style items built from MCQs not in LongReason (ReClor, LogiQA 2.0).

Each passage was expanded by Qwen3.8-27B (`scripts/gen_expanded.py`) and the paragraphs are scattered through DCLM
filler at 128k. Rows, prompt and scoring are `longreason_real`'s; sub-calls go to a real model.
``split``: "train" (all but every 10th item) or "val" (every 10th).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from datasets import Dataset

import rlm_train
from longreason_real.env import _option_letters, _score
from .filler import LENGTH_CHARS, assemble, sample_filler
from rlm_train.frozen import frozen_path

ITEMS = Path(os.environ.get("LONGREASON_EXPANDED_ITEMS") or Path(__file__).resolve().parent / "data" / "expanded_items.jsonl")
ROOT_PREFIX = ("Answer the multiple-choice question below. The scenario it refers to is embedded somewhere inside the long `context` "
               "passage, surrounded by unrelated filler text — locate it first, then reason.\n\n")
ROOT_SUFFIX = "\n\nRespond with the single option letter only."


def build_context(item: dict, length: str = "128k", seed: int = 0) -> str:
    rng = random.Random(f"longreason-expanded:{item['id']}:{length}:{seed}")
    paras = item["paragraphs"]
    target = int(LENGTH_CHARS[length] * rng.uniform(0.97, 1.03))
    filler = sample_filler(rng, max(2_000, target - sum(len(p) + 2 for p in paras)))
    return assemble(rng, filler, paras, [])


def _rows(items: list[dict], length: str) -> list[dict]:
    rows = []
    for i, it in enumerate(items):
        inquiry = it["question"].strip()
        info = {"context": build_context(it, length), "root_prompt": ROOT_PREFIX + inquiry + ROOT_SUFFIX,
                "gold_letter": it["answer"], "option_letters": _option_letters(inquiry), "longreason_split": f"expanded-{length}",
                "example_idx": it["id"], "source": it["source"]}
        rows.append({"example_id": i, "prompt": [{"role": "user", "content": "Answer the multiple-choice question."}],
                     "answer": it["answer"], "info": json.dumps(info)})
    return rows


def load_environment(*, items_path: str | None = None, split: str = "train", num_examples: int = -1, length: str = "128k",
                     max_iterations: int = 20, sub_max_tokens: int = 8192, min_subcall: int = 1, **kwargs: Any):
    """``split``: "train" = all items except every 10th, "val" = every 10th (by file order)."""
    items = [json.loads(l) for l in open(items_path or frozen_path(ITEMS, "longreason/expanded_items.jsonl"))]
    items = [it for k, it in enumerate(items) if (k % 10 == 9) == (split == "val")]
    if num_examples and num_examples > 0:
        items = items[:num_examples]
    rubric = rlm_train.RLMTrainRubric(correctness=_score, weight=1.0, min_subcall=min_subcall)
    kwargs.setdefault("code_block_feedback", True)  # Section 4 setting: feedback on turns without a code block
    return rlm_train.RLMTrainEnv(dataset=Dataset.from_list(_rows(items, length)), max_iterations=max_iterations,
                                 sub_sampling_args={"max_tokens": sub_max_tokens}, rubric=rubric, **kwargs)


__all__ = ["build_context", "load_environment"]
