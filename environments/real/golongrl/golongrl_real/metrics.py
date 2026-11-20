"""GoLongRL reward functions (table and summary families), ported from upstream ``xiaoxuanNLP/GoLongRL``.

Upstream's ``<think>`` format gate is not applied. ROUGE-L is an LCS F1, with jieba segmentation for Chinese if available.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from typing import Any


def fix_space(text: str) -> str:
    return " ".join(text.split())


def normalize_answers(answers: list[str]) -> list[str]:
    return [fix_space(a.lower().strip()) for a in answers]


def normalize_prediction_abstract(abstract: str) -> str:
    return fix_space(abstract.lower().strip())


def _segment(text: str, is_zh: bool) -> list[str]:
    if not is_zh:
        return text.split()
    try:
        import jieba  # type: ignore

        return [t for t in jieba.cut(text, cut_all=False) if t.strip()]
    except Exception:  # pragma: no cover
        return [c for c in text if not c.isspace()]


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(reference: str, prediction: str, is_zh: bool = False) -> float:
    ref, pred = _segment(reference, is_zh), _segment(prediction, is_zh)
    if not ref or not pred:
        return 0.0
    lcs = _lcs_len(ref, pred)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def summary_score(answers: list[str], prediction: str, is_zh: bool) -> float:
    """Max ROUGE-L F over the reference summaries (upstream ``Summary``)."""
    a = normalize_answers(answers)
    p = normalize_prediction_abstract(prediction)
    if not a or not p:
        return 0.0
    return max(rouge_l(ref, p, is_zh) for ref in a)


TASK_METRIC_CONFIG: dict[str, str] = {
    "T4.1 Global-Coverage Constrained Summary": "Summary",
    "T4.2 Query-Focused Summary": "Summary",
}


def longbench_pro_score(
    model_output: str,
    ground_truth: list[str],
    sub_task: str,
    language: str = "English",
) -> float:
    if not model_output or not model_output.strip():
        return 0.0
    if not ground_truth:
        return 0.0
    metric = TASK_METRIC_CONFIG.get(sub_task)
    if not metric:
        raise ValueError(f"Unknown LongBench-Pro sub_task: {sub_task!r}")
    is_zh = language == "Chinese"
    try:
        if metric == "Summary":
            score = summary_score(ground_truth, model_output, is_zh)
        else:  # pragma: no cover
            raise ValueError(f"Unknown metric: {metric}")
        return float(min(max(score, 0.0), 1.0))
    except Exception:  # pragma: no cover  (upstream swallows these too)
        return 0.0


def _strip_answer_tag(text: str) -> str:
    if "<answer>" in text:
        return text.partition("<answer>")[2].partition("</answer>")[0].strip()
    return text.strip()


def multitableqa_score(solution_str: str, golden_label: str) -> float:
    """``multitableqa.compute_score``: IoU over the multiset of result rows."""
    try:
        answer = json.loads(golden_label)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    raw = _strip_answer_tag(solution_str or "")
    pred: Any = []
    for loader in (json.loads, ast.literal_eval):
        try:
            pred = loader(raw).get("data", [])
            break
        except Exception:  # noqa: BLE001
            pred = []
    if not isinstance(pred, list):
        return 0.0

    def to_key(r: Any) -> Any:
        return tuple(str(v) for v in r) if isinstance(r, list) else str(r)

    ans_list = answer.get("data", []) if isinstance(answer, dict) else []
    pc, gc = Counter(to_key(r) for r in pred), Counter(to_key(r) for r in ans_list)
    matches = sum((pc & gc).values())
    total = len(pred) + len(ans_list) - matches
    return matches / total if total > 0 else 0.0


def metric_name(data_source: str, reward_mode: str) -> str:
    if data_source == "multitableqa_pretraining":
        return "iou"
    if data_source == "longbench_pro":
        return {
            "Summary": "rouge_l",
        }[TASK_METRIC_CONFIG[reward_mode]]
    raise ValueError(f"Unknown data_source {data_source!r}")


def compute_score(
    *,
    data_source: str,
    reward_mode: str,
    prediction: str,
    doc_ids: list[str] | None = None,
    golden_label: str = "",
    language: str = "",
) -> float:
    """Score one row, like upstream ``default_compute_score`` without the ``<think>`` gate."""
    text = (prediction or "")
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if data_source == "multitableqa_pretraining":
        return multitableqa_score(text, golden_label)
    if data_source == "longbench_pro":
        return longbench_pro_score(
            text, list(doc_ids or []), reward_mode, language or "English"
        )
    raise ValueError(f"Unknown data_source {data_source!r}")


def gold_answer_text(
    *, data_source: str, reward_mode: str, doc_ids: list[str] | None, golden_label: str
) -> str:
    """The answer string that scores 1.0."""
    docs = list(doc_ids or [])
    if data_source == "multitableqa_pretraining":
        return f"<answer>{golden_label}</answer>"
    if data_source == "longbench_pro":
        if TASK_METRIC_CONFIG[reward_mode] == "Summary":
            return docs[0] if docs else ""
        return "\n".join(docs)
    raise ValueError(f"Unknown data_source {data_source!r}")


__all__ = [
    "TASK_METRIC_CONFIG",
    "compute_score",
    "gold_answer_text",
    "longbench_pro_score",
    "metric_name",
    "multitableqa_score",
    "rouge_l",
    "summary_score",
]
