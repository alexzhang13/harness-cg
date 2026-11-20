"""OOLONG-Pairs (`oolong-pairs-real`): list every pair of users whose hidden label histories meet a condition.

20 questions per context length over the OOLONG-synth ``trec_coarse`` window. Reward is F1 over the pair sets.
"""

from __future__ import annotations

import os
import json
import random
import re
from importlib.resources import files
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

import rlm_train

# pinned dataset revision (later revisions change the context text)
OOLONG_SYNTH_REVISION = os.environ.get("OOLONG_SYNTH_REVISION", "f0d59eaf0febf130664cfceb710436c8e3216b2b")

_DATA_DIR = files("oolong_pairs_real") / "data"
_PAIRS_REPO = "mit-oasys/oolong-pairs"
_PAIRS_REVISION = "d1e1522b86ac0c169bbc890b0471408aaa29e8fa"
SUPPORTED_CONTEXT_LENGTHS = (4096, 8192, 16384, 32768)


_PAIR_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_BARE_PAIR_RE = re.compile(r"\b(-?\d+)\s*[,/&]\s*(-?\d+)\b")
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _parse_pairs(text: str) -> set[tuple[int, int]]:
    """Pairs written as ``(a, b)``; if there are none, bare ``a, b`` / ``a/b`` / ``a&b``. Order-normalised."""
    pairs: set[tuple[int, int]] = set()
    matches = _PAIR_RE.findall(text) or _BARE_PAIR_RE.findall(text)
    for a, b in matches:
        ia, ib = int(a), int(b)
        pairs.add((ia, ib) if ia < ib else (ib, ia))
    return pairs


def _score_f1(predicted: str, gold_list: list[str]) -> float:
    """F1 over predicted vs gold pair sets."""
    cleaned = _THINK_RE.sub("", predicted or "")
    gt: set[tuple[int, int]] = set()
    for g in gold_list:
        gt |= _parse_pairs(str(g))
    pred = _parse_pairs(cleaned)
    if not gt and not pred:
        return 1.0
    if not pred or not gt:
        return 0.0
    correct = len(pred & gt)
    if correct == 0:
        return 0.0
    p = correct / len(pred)
    r = correct / len(gt)
    return 2 * p * r / (p + r)


async def _reward(answer: str, state: vf.State, **_kw: Any) -> float:
    final = state.get("rlm_final_answer") or ""
    try:
        gold = json.loads(answer)
    except Exception:  # noqa: BLE001
        gold = [answer]
    return _score_f1(final, gold if isinstance(gold, list) else [gold])


def _load_pairs(context_length: int) -> list[dict]:
    """The 20 questions (``question``, ``answer`` = list of ``"(a, b)"`` strings) for one context length."""
    name = f"oolong-pairs-{context_length}.json"
    path = _DATA_DIR / name
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(_PAIRS_REPO, f"data/{name}", repo_type="dataset", revision=_PAIRS_REVISION))
    return json.loads(path.read_text())


def _load_trec_coarse_context(context_length: int) -> tuple[str, str]:
    """(unlabeled, labeled) text of the ``trec_coarse`` window with this ``context_len``."""
    ds = load_dataset("oolongbench/oolong-synth", revision=OOLONG_SYNTH_REVISION, split="validation")
    for ex in ds:
        if ex.get("dataset") == "trec_coarse" and ex.get("context_len") == context_length:
            return str(ex["context_window_text"]), str(ex["context_window_text_with_labels"])
    raise FileNotFoundError(f"No trec_coarse example at context_len={context_length}")


def load_environment(
    *,
    context_length: int = 32768,
    num_examples: int = 20,
    shuffle: bool = True,
    seed: int | None = 42,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_iterations: int = 2,
    min_subcall: int = 1,
    max_sub_calls: int = 500,
    **kwargs: Any,
) -> vf.Environment:
    if context_length not in SUPPORTED_CONTEXT_LENGTHS:
        raise ValueError(
            f"context_length={context_length!r} not bundled. "
            f"Available: {SUPPORTED_CONTEXT_LENGTHS}"
        )

    questions = _load_pairs(context_length)
    if shuffle:
        rng = random.Random(seed if seed is not None else 0)
        rng.shuffle(questions)
    if num_examples > 0:
        questions = questions[:num_examples]

    unlabeled, _labeled = _load_trec_coarse_context(context_length)

    rows: list[dict] = []
    for idx, q in enumerate(questions):
        raw_question = q["question"]
        gold = q.get("answer", []) or []
        rows.append(
            {
                "example_id": idx,
                "prompt": [{"role": "user", "content": raw_question}],
                "answer": json.dumps(list(gold)),
                "info": {
                    "context": unlabeled,
                    "raw_question": raw_question,
                    "context_len": context_length,
                    "dataset": "oolong_pairs_trec_coarse_real",
                    "root_prompt": raw_question,
                },
            }
        )

    dataset = Dataset.from_list(rows)
    rubric = rlm_train.RLMTrainRubric(
        correctness=_reward, weight=1.0, min_iterations=min_iterations,
        min_subcall=min_subcall,
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        max_sub_calls=max_sub_calls,
        **kwargs,
    )


__all__ = ["load_environment"]
