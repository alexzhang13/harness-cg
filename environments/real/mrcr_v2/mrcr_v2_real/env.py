"""MRCR v2 (`mrcr-v2-real`): reproduce one earlier assistant reply from a long multi-turn transcript.

Reward is the SequenceMatcher ratio between gold and answer after the 12-character hash prefix (0 without the hash).
"""

from __future__ import annotations

import csv
import difflib
import io
import json
import os
import random
import sys
import urllib.error
import urllib.request
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train

_BASE_URL = "https://storage.googleapis.com/mrcr_v2"

# context_range -> token-count bucket in the CSV file name
_RANGE_PATTERNS = {
    "32k-64k": "in_(32768,65536)",
    "1m-2m": "in_(1048576,2097152)",
    "2m-4m": "in_(2097152,4194304)",
}

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "rlm_train_real", "mrcr_v2")


def _csv_filename(needle_count: int, ctx_range: str) -> str:
    pat = _RANGE_PATTERNS[ctx_range.lower().replace(" ", "")]
    return f"mrcr_v2p1_{needle_count}needle_{pat}_dynamic_fewshot_text_style_fast.csv"


def _parse_row(r: dict) -> dict:
    question = r["view_ops"].strip()
    if question.startswith("User: "):
        question = question[len("User: "):]
    if question.endswith("Assistant:"):
        question = question[:-len("Assistant:")].strip()
    return {
        "question": question,
        "context": r["queries"],
        "answer": r["answer"],
        "context_len": int(r["context_len"]),
    }


def _load_csv(needle_count: int, ctx_range: str, max_n: int | None) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    name = _csv_filename(needle_count, ctx_range)
    path = os.path.join(_CACHE_DIR, name)
    url = f"{_BASE_URL}/{name}"

    # cached file
    if os.path.exists(path) and os.path.getsize(path) > 0:
        out: list[dict] = []
        with open(path, newline="") as f:
            for i, r in enumerate(csv.DictReader(f)):
                if max_n is not None and i >= max_n:
                    break
                out.append(_parse_row(r))
        return out

    # with a row cap, stream the first rows without caching (long buckets are several GB)
    if max_n is not None:
        try:
            resp = urllib.request.urlopen(url)
        except urllib.error.HTTPError as e:
            raise FileNotFoundError(
                f"Failed to fetch MRCR v2 CSV {url} (HTTP {e.code})"
            ) from e
        with resp:
            reader = csv.DictReader(io.TextIOWrapper(resp, encoding="utf-8", newline=""))
            out = []
            for i, r in enumerate(reader):
                if i >= max_n:
                    break
                out.append(_parse_row(r))
        return out

    # no cap: download the whole file
    tmp = path + ".partial"
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.rename(tmp, path)
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.append(_parse_row(r))
    return out


def _mrcr_score(prediction: str, target: str) -> float:
    """SequenceMatcher ratio on the text after the 12-character hash (last occurrence in the prediction)."""
    if not prediction or not target or len(target) < 12:
        return 0.0
    h = target[:12]
    target_ref = target[12:].strip()
    pred = prediction.strip()
    start = pred.rfind(h)
    if start == -1:
        return 0.0
    pred_after = pred[start + 12:].strip()
    return difflib.SequenceMatcher(a=target_ref, b=pred_after).ratio()


async def _score(answer: str, state: vf.State, **_kw: Any) -> float:
    final = state.get("rlm_final_answer") or state.get("final_answer") or ""
    return _mrcr_score(final, answer)


_PREAMBLE = (
    "You are given a long multi-turn conversation transcript between a user and "
    "an assistant. Each turn the user requests a piece of writing with a specific "
    "format/style/topic; the assistant responds with a distinct piece of writing. "
    "Your task: search the transcript, locate the assistant response matching the "
    "description, and reproduce it EXACTLY — prepended by the provided 12-char hash."
)


def _build_dataset(
    *, needle_count: int, context_range: str, num_examples: int, seed: int | None
) -> Dataset:
    max_n = num_examples if num_examples > 0 else None
    examples = _load_csv(needle_count, context_range, max_n)
    if not examples:
        raise ValueError(f"No MRCR v2 examples loaded for {needle_count}/{context_range}")
    if seed is not None:
        random.Random(seed).shuffle(examples)
    rows: list[dict] = []
    for i, ex in enumerate(examples):
        info = {
            "context": ex["context"],
            "root_prompt": (
                f"{_PREAMBLE}\n\n{ex['question']}\n"
                f"(The transcript is in `context`. Answer with the 12-char "
                f"hash followed by the reproduced response.)"
            ),
            "context_len": ex["context_len"],
        }
        rows.append({
            "example_id": i,
            "prompt": [{"role": "user", "content": ex["question"]}],
            "answer": ex["answer"],
            "info": json.dumps(info),
        })
    return Dataset.from_list(rows)


def load_environment(
    *,
    needle_count: int = 2,
    context_range: str = "32k-64k",
    num_examples: int = -1,
    seed: int | None = 42,
    max_iterations: int = 12,
    sub_max_tokens: int = 4096,
    min_subcall: int = 1,
    **kwargs: Any,
) -> vf.Environment:
    dataset = _build_dataset(
        needle_count=needle_count, context_range=context_range,
        num_examples=num_examples, seed=seed,
    )
    rubric = rlm_train.RLMTrainRubric(correctness=_score, weight=1.0, min_subcall=min_subcall)
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = ["load_environment"]
