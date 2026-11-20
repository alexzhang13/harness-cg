"""LongProc ToM tracking (`longproc-real`): write the belief trace for a long story.

With ``tom_dense_reward=True`` the reward is per-line accuracy; the strict upstream metric is logged alongside.
"""

from __future__ import annotations

import json
import re
import string
from typing import Any

import verifiers as vf
from datasets import Dataset, concatenate_datasets, load_dataset

import rlm_train


SUPPORTED_TASKS = ("tom_tracking",)
SUPPORTED_LENGTHS = ("2k", "8k")


def _normalize_tom(s: str) -> str:
    """Lowercase, strip punctuation, articles, and collapse whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(
            r"\b(a|an|the|on|in|at|the|step|thinks|think|believes|believe|is|are|of|location|know|knows|belief)\b",
            " ",
            text,
        )

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punctuation(text: str) -> str:
        return "".join(ch for ch in text if ch not in string.punctuation and ch != "’")

    return white_space_fix(remove_articles(remove_punctuation(s.lower())))


def _extract_belief_content(line: str) -> str | None:
    if line.startswith("-"):
        return _normalize_tom(line.split("-", 1)[1].strip())
    return None


def _tom_beliefs(prediction: str, gt_text: str) -> tuple[list[str], list[str]]:
    model_beliefs = [b for line in prediction.strip().split("\n")
                     if (b := _extract_belief_content(line)) is not None]
    gt_beliefs = [b for line in gt_text.strip().split("\n")
                  if (b := _extract_belief_content(line)) is not None]
    return model_beliefs, gt_beliefs


def _score_tom_tracking_strict(prediction: str, meta: dict) -> float:
    """Upstream metric: fraction of belief lines correct before the first mismatch."""
    gt_text = meta.get("ground_truth_tom") or meta.get("ground_truth") or ""
    if not gt_text:
        return 0.0
    model_beliefs, gt_beliefs = _tom_beliefs(prediction, gt_text)
    if not gt_beliefs:
        return 0.0
    if len(model_beliefs) == len(gt_beliefs) and all(a == b for a, b in zip(model_beliefs, gt_beliefs, strict=False)):
        return 1.0
    first_diff = next((i for i, (a, b) in enumerate(zip(model_beliefs, gt_beliefs, strict=False)) if a != b), None)
    if first_diff is not None:
        return first_diff / len(gt_beliefs)
    return min(len(model_beliefs), len(gt_beliefs)) / len(gt_beliefs)


def _score_tom_tracking_dense(prediction: str, meta: dict) -> float:
    """Fraction of reference lines matched at the same index (denominator: the longer line count)."""
    gt_text = meta.get("ground_truth_tom") or meta.get("ground_truth") or ""
    if not gt_text:
        return 0.0
    model_beliefs, gt_beliefs = _tom_beliefs(prediction, gt_text)
    if not gt_beliefs:
        return 0.0
    correct = sum(1 for i, g in enumerate(gt_beliefs)
                  if i < len(model_beliefs) and model_beliefs[i] == g)
    return correct / max(len(gt_beliefs), len(model_beliefs))


_TASK_SCORERS = {
    "tom_tracking": _score_tom_tracking_strict,
}


def _make_score(tom_dense: bool):
    """Reward fn: dense per-line score if ``tom_dense``, else the strict metric."""
    async def correctness(info, state: vf.State, **_kw: Any) -> float:
        meta = json.loads(info) if isinstance(info, str) else info
        task = meta.get("longproc_task") or "tom_tracking"
        if task == "tom_tracking" and tom_dense:
            fn = _score_tom_tracking_dense
        else:
            fn = _TASK_SCORERS.get(task)
        if fn is None:
            return 0.0
        final = state.get("rlm_final_answer") or state.get("final_answer") or ""
        if not isinstance(final, str):
            final = str(final)
        try:
            return float(fn(final, meta))
        except Exception:
            return 0.0

    return correctness


def _make_tom_metric(*, dense: bool):
    """Weight-0 metric for one of the two scorers, so both are always logged."""
    fn = _score_tom_tracking_dense if dense else _score_tom_tracking_strict

    async def _metric(info, state: vf.State, **_kw: Any) -> float:
        meta = json.loads(info) if isinstance(info, str) else info
        if (meta.get("longproc_task") or "") != "tom_tracking":
            return 0.0
        final = state.get("rlm_final_answer") or state.get("final_answer") or ""
        if not isinstance(final, str):
            final = str(final)
        try:
            return float(fn(final, meta))
        except Exception:
            return 0.0

    _metric.__name__ = "tom_dense" if dense else "tom_strict"
    return _metric


def _build_dataset(*, tasks: list[str], length: str, num_examples: int, seed: int | None) -> Dataset:
    if length not in SUPPORTED_LENGTHS:
        raise ValueError(f"Unsupported LongProc length tier: {length!r}. "
                         f"Choose from {SUPPORTED_LENGTHS}")
    parts: list[Dataset] = []
    for task in tasks:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported LongProc task: {task!r}. "
                             f"Choose from {SUPPORTED_TASKS}")
        try:
            ds = load_dataset("PrincetonPLI/LongProc", f"{task}_{length}", split="test")
        except Exception as e:
            raise ValueError(f"LongProc config {task}_{length!r} not available upstream: {e}") from e
        parts.append(ds.map(
            lambda r, _t=task: {"_longproc_task": _t},
            load_from_cache_file=False,
        ))
    big = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if seed is not None:
        big = big.shuffle(seed=seed)
    if num_examples and num_examples > 0:
        big = big.select(range(min(num_examples, len(big))))

    out: list[dict] = []
    for i, r in enumerate(big):
        task = r["_longproc_task"]
        meta_raw = r.get("metadata") or "{}"
        try:
            task_meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
        except json.JSONDecodeError:
            task_meta = {}
        if task == "tom_tracking":
            task_meta["ground_truth_tom"] = r.get("reference_output") or ""
        prompt_text = r.get("input_prompt") or ""
        info = {
            "context": prompt_text,
            "root_prompt": (
                "Read the procedural-generation prompt in `context` and produce "
                "the full structured long-form output it asks for (including the "
                "tagged region the prompt specifies)."
            ),
            "longproc_task": task,
            **task_meta,
        }
        out.append({
            "example_id": i,
            "prompt": [{"role": "user", "content": prompt_text[:8000]}],
            "answer": r.get("reference_output") or "",
            "info": json.dumps(info),
        })
    return Dataset.from_list(out)


def load_environment(
    *,
    tasks: str | list[str] = "tom_tracking",
    length: str = "8k",
    num_examples: int = -1,
    seed: int = 42,
    max_iterations: int = 12,
    sub_max_tokens: int = 8192,
    min_subcall: int = 0,
    tom_dense_reward: bool = False,
    **kwargs: Any,
) -> vf.Environment:
    """Load a LongProc length tier wired through RLMTrainEnv."""
    if isinstance(tasks, str):
        task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    else:
        task_list = list(tasks)
    if not task_list:
        raise ValueError("longproc-real: empty `tasks`.")
    dataset = _build_dataset(tasks=task_list, length=length, num_examples=num_examples, seed=seed)
    rubric = rlm_train.RLMTrainRubric(
        correctness=_make_score(tom_dense_reward), weight=1.0, min_subcall=min_subcall,
    )
    if "tom_tracking" in task_list:
        rubric.add_metric(_make_tom_metric(dense=True))
        rubric.add_metric(_make_tom_metric(dense=False))
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = ["load_environment"]
