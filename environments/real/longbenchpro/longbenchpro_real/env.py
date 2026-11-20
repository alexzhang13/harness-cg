"""LongBench-Pro (`longbenchpro-real`), scored with the benchmark's per-task metric on the text after ``[Answer]``.

Summarization tasks (T4.*) are excluded since their metric needs an embedding model.
"""

from __future__ import annotations

import json
import random
import re
from itertools import combinations
from typing import Any

import pytrec_eval
import verifiers as vf
from datasets import Dataset, load_dataset

import rlm_train


# Per-task metrics, as in the upstream LongBench-Pro evaluation code.
def _fix(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _normalize_pred(pred: str) -> list[str]:
    if "[Answer]" in pred:
        pred = pred[pred.rfind("[Answer]") + len("[Answer]"):]
    elif "[答案]" in pred:
        pred = pred[pred.rfind("[答案]") + len("[答案]"):]
    pred = pred.lower()
    return [s for line in pred.split("\n") if (s := _fix(line.strip()))]


def _normalize_answers(answers):
    return [_fix(a.lower().strip()) for a in answers]


def _accuracy(answers, pred):
    a, p = _normalize_answers(answers), _normalize_pred(pred)
    return 0.0 if not a or not p else (1.0 if a[0] == p[0] else 0.0)


def _f1(answers, pred):
    a, p = set(_normalize_answers(answers)), set(_normalize_pred(pred))
    common = a & p
    if not common or not p or not a:
        return 0.0
    P, R = len(common) / len(p), len(common) / len(a)
    return 0.0 if P + R == 0 else 2 * P * R / (P + R)


def _sub_em(answers, pred):
    a, p = _normalize_answers(answers), _normalize_pred(pred)
    if not a or not p:
        return 0.0
    found = sum(1.0 for x in a if x in p)
    return found / len(a)


def _ndcg(answers, pred):
    a, p = _normalize_answers(answers), _normalize_pred(pred)
    k = len(a)
    if k == 0 or not p:
        return 0.0
    qrel = {"query": {x: len(a) - i for i, x in enumerate(a)}}
    run = {"query": {x: len(p) - i for i, x in enumerate(p)}}
    ev = pytrec_eval.RelevanceEvaluator(qrel, {f"ndcg_cut.{k}"})
    scores = ev.evaluate(run)
    return sum(s[f"ndcg_cut_{k}"] for s in scores.values()) / len(scores)


def _pairwise(answers, pred):
    a, p = _normalize_answers(answers), _normalize_pred(pred)
    if len(a) < 2 or len(p) < 2:
        return 0.0
    total = len(p) * (len(p) - 1) // 2
    ix = {x: i for i, x in enumerate(p)}
    n_correct = 0
    for x, y in combinations(a, 2):
        if x in ix and y in ix and ix[x] < ix[y]:
            n_correct += 1
    return n_correct / total


# secondary_task prefix -> metric
_TASK_METRIC = {
    "T1.1": "ndcg", "T1.2": "ndcg",
    "T2.1": "pairwise_accuracy", "T2.2": "pairwise_accuracy",
    "T3.1": "accuracy", "T3.2": "accuracy",
    "T5.1": "f1_score", "T5.2": "f1_score",
    "T6.1": "sub_em", "T6.2": "f1_score", "T6.3": "pairwise_accuracy",
    "T7.1": "f1_score", "T7.2": "f1_score", "T7.3": "f1_score",
    "T8.1": "sub_em", "T8.2": "sub_em", "T8.3": "sub_em",
    "T9.1": "f1_score", "T9.2": "f1_score",
    "T10.1": "sub_em", "T10.2": "sub_em",
    "T11.1": "accuracy", "T11.2": "accuracy",
}
_METRIC_FN = {
    "accuracy": _accuracy, "f1_score": _f1, "sub_em": _sub_em,
    "ndcg": _ndcg, "pairwise_accuracy": _pairwise,
}


def _compute(sec_task: str, answers: list[str], pred: str) -> float:
    prefix = sec_task.split(" ")[0] if " " in sec_task else sec_task
    fn = _METRIC_FN[_TASK_METRIC.get(prefix, "accuracy")]
    return max(0.0, min(1.0, fn(answers, pred)))


async def _score(info, state: vf.State, **_kw: Any) -> float:
    final = state.get("rlm_final_answer") or state.get("final_answer") or ""
    meta = json.loads(info) if isinstance(info, str) else info
    return _compute(meta.get("secondary_task", ""), meta.get("answers", []), final)


# Summarization: its metric needs an embedding model.
_EXCLUDED_PREFIXES = ("T4.",)


def _build_dataset(
    *, language: str, token_length: str, thinking: bool, num_examples: int, seed: int | None,
) -> Dataset:
    question_col = "question_thinking" if thinking else "question_nonthinking"

    def _matches(ex):
        if any(ex["secondary_task"].startswith(p) for p in _EXCLUDED_PREFIXES):
            return False
        if language != "all" and ex["language"] != language:
            return False
        if token_length != "all" and ex["token_length"] != token_length:
            return False
        return True

    # with a cap take the first matches of the shuffled stream, else take all and shuffle
    if num_examples > 0:
        stream = load_dataset("caskcsg/LongBench-Pro", split="test", streaming=True)
        if seed is not None:
            stream = stream.shuffle(seed=seed, buffer_size=10_000)
        samples: list[dict] = []
        for ex in stream:
            if _matches(ex):
                samples.append(ex)
                if len(samples) >= num_examples:
                    break
    else:
        raw = load_dataset("caskcsg/LongBench-Pro", split="test")
        samples = [ex for ex in raw if _matches(ex)]
        if seed is not None:
            random.Random(seed).shuffle(samples)

    rows: list[dict] = []
    for i, ex in enumerate(samples):
        q = ex[question_col]
        info = {
            "context": ex["context"],
            "root_prompt": q,
            "secondary_task": ex["secondary_task"],
            "answers": ex["answer"],
        }
        rows.append({
            "example_id": i,
            "prompt": [{"role": "user", "content": q}],
            "answer": json.dumps(ex["answer"]),
            "info": json.dumps(info),
        })
    return Dataset.from_list(rows)


def load_environment(
    *,
    language: str = "all",
    token_length: str = "all",
    thinking: bool = False,
    num_examples: int = -1,
    seed: int | None = 42,
    max_iterations: int = 12,
    sub_max_tokens: int = 4096,
    min_iterations: int = 2,
    min_subcall: int = 0,
    **kwargs: Any,
) -> vf.Environment:
    dataset = _build_dataset(
        language=language, token_length=token_length, thinking=thinking,
        num_examples=num_examples, seed=seed,
    )
    rubric = rlm_train.RLMTrainRubric(
        correctness=_score, weight=1.0,
        min_iterations=min_iterations,
        min_subcall=min_subcall,
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = ["load_environment"]
