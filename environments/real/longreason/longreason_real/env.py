"""LongReason (`longreason-real`): multiple-choice reasoning over a scenario hidden in long filler text.

One HF split per length (`8k` ... `128k`); reward is exact match of the option letter.
"""

from __future__ import annotations

import json
import re
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

import rlm_train

HF_DATASET = "lz1bytedance/LongReason"

# HF splits: the length-controlled ones, plus the unexpanded source question and its intermediate rewrite.
LENGTH_SPLITS = ("8k", "16k", "32k", "64k", "128k")
ALL_SPLITS = ("original", "expanded", *LENGTH_SPLITS)

_LETTER_RE = re.compile(r"^\s*([A-H])[.)\s]", re.MULTILINE)


def _option_letters(inquiry: str) -> list[str]:
    """The option letters actually present in the question text."""
    return sorted(set(_LETTER_RE.findall(inquiry or "")))


def _parse_letter(s: str, allowed: list[str] | None = None) -> str | None:
    """Pull one option letter out of a free-form answer (explicit "answer is X" forms first, then a lone capital)."""
    if not s:
        return None
    allowed = allowed or list("ABCDEFGH")
    cls = "".join(allowed)
    s = s.strip().strip("`'\" ")
    if len(s) == 1 and s.upper() in allowed:
        return s.upper()
    stated = re.findall(rf"(?i:answer|choice|option)\s+(?i:is|would be|should be)\s*[:(\[]?\s*([{cls}])\b", s)
    if stated:
        return stated[-1]
    for pat in (
        rf"(?:answer|letter|choice|option|final)[^A-Za-z]*([{cls}])\b",
        rf"\b([{cls}])\)",
        rf"\(([{cls}])\)",
        rf"\[([{cls}])\]",
    ):
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.search(rf"\b([{cls}])\b", s)
    if m:
        return m.group(1).upper()
    return None


async def _score(info, state: vf.State, **_kw: Any) -> float:
    meta = json.loads(info) if isinstance(info, str) else info
    final = state.get("rlm_final_answer") or state.get("final_answer") or ""
    allowed = meta.get("option_letters") or None
    pred = _parse_letter(final if isinstance(final, str) else str(final), allowed)
    gold = (meta.get("gold_letter") or "").strip().upper()
    return 1.0 if (pred and gold and pred == gold) else 0.0


def _build_dataset(*, split: str, num_examples: int, seed: int | None) -> Dataset:
    ds = load_dataset(HF_DATASET, split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if num_examples and num_examples > 0:
        ds = ds.select(range(min(num_examples, len(ds))))

    rows: list[dict] = []
    for i, r in enumerate(ds):
        background = (r.get("background") or "").strip()
        inquiry = (r.get("final_inquiry") or "").strip()
        gold = (r.get("answer") or "").strip().upper()
        if not background or not inquiry or not gold:   # unusable row
            continue
        letters = _option_letters(inquiry)
        info = {
            "context": background,
            "root_prompt": (
                "Answer the multiple-choice question below. The scenario it "
                "refers to is embedded somewhere inside the long `context` "
                "passage, surrounded by unrelated filler text — locate it "
                "first, then reason.\n\n"
                f"{inquiry}\n\n"
                "Respond with the single option letter only."
            ),
            "gold_letter": gold,
            "option_letters": letters,
            "longreason_split": split,
            "example_idx": r.get("example_idx") or "",
        }
        rows.append({
            "example_id": i,
            "prompt": [{"role": "user", "content": "Answer the multiple-choice question."}],
            "answer": gold,
            "info": json.dumps(info),
        })
    return Dataset.from_list(rows)


def load_environment(
    *,
    split: str = "128k",
    num_examples: int = -1,
    seed: int = 42,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_subcall: int = 1,
    **kwargs: Any,
) -> vf.Environment:
    """Load LongReason as an RLMTrainEnv. split: one of ``ALL_SPLITS``."""
    if split not in ALL_SPLITS:
        raise ValueError(f"split={split!r} not available. Choose from {ALL_SPLITS}.")
    dataset = _build_dataset(split=split, num_examples=num_examples, seed=seed)
    rubric = rlm_train.RLMTrainRubric(
        correctness=_score, weight=1.0, min_subcall=min_subcall,
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = ["load_environment", "LENGTH_SPLITS", "ALL_SPLITS"]
