"""OOLONG-synth (`oolong-real`): aggregate questions over unlabeled ``Date || User || Instance`` lines.

Task families ``trec_coarse`` and ``spam``. Only the final answer is graded (text after the last ``Answer:`` marker,
else the last line), following the reference OOLONG scorer.
"""

from __future__ import annotations

import os
import ast
import json
import random
import re
from datetime import datetime
from typing import Any

import dateutil.parser
import verifiers as vf
from datasets import Dataset, load_dataset

import rlm_train

# pinned dataset revision (later revisions change the context text)
OOLONG_SYNTH_REVISION = os.environ.get("OOLONG_SYNTH_REVISION", "f0d59eaf0febf130664cfceb710436c8e3216b2b")


# Grader: the reference OOLONG scorer (abertsch72/oolong eval_helpers.py), applied to the extracted final answer.

COMPARISON_PHRASES = ("more common than", "less common than", "same frequency as")


def _find_comparison_phrase(output: str) -> str | None:
    """Return the last-occurring canonical comparison phrase in ``output``."""
    out_low = output.lower()
    hits = [(out_low.rfind(p), p) for p in COMPARISON_PHRASES if p in out_low]
    return max(hits)[1] if hits else None


_ANSWER_MARKER = re.compile(r"(?:final\s+answer|answer|label)\s*:", re.IGNORECASE)


def _extract_final_answer(output: str) -> str:
    """First line after the last ``Answer:`` / ``Label:`` / ``Final Answer:`` marker, else the last non-empty line."""
    text = (output or "").strip()
    if not text:
        return ""
    hits = list(_ANSWER_MARKER.finditer(text))
    if hits:
        tail = text[hits[-1].end():].strip()
        cand = tail.splitlines()[0].strip() if tail else ""
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cand = lines[-1] if lines else text
    return cand.replace("*", "").replace("[", "").replace("]", "").strip()


# Label and user answers must commit to one value (no hedging across labels / ids); otherwise fall back to containment.

# label set per task family, when the question doesn't list it
_LABEL_SETS: dict[str, tuple[str, ...]] = {
    "trec_coarse": (
        "numeric value", "entity", "location",
        "description and abstract concept", "abbreviation", "human being",
    ),
    "spam": ("spam", "ham"),
}

# "... where answer is one of the labels: ham, spam." in the question gives the closed set for that row.
_LABEL_LIST_RE = re.compile(r"one of the labels:\s*([^.?\n]+)", re.IGNORECASE)


def _gold_labels(answer: str) -> list[str]:
    """Full parsed gold list, lowercased (the gold may be a tie of 2-3 labels)."""
    try:
        vals = ast.literal_eval(answer)
    except Exception:
        return []
    if not isinstance(vals, (list, tuple)):
        return []
    return [str(v).strip().lower() for v in vals if str(v).strip()]


def _closed_label_set(datapoint: dict, gold_s: str) -> tuple[str, ...]:
    """The task's closed label set, or () if it cannot be established."""
    gold_l = gold_s.strip().lower()
    m = _LABEL_LIST_RE.search(str(datapoint.get("root_prompt", "")))
    if m:
        parsed = tuple(
            x for x in (p.strip().lower() for p in m.group(1).split(","))
            if x
        )
        if gold_l in parsed:
            return parsed
    fixed = _LABEL_SETS.get(str(datapoint.get("dataset", "")), ())
    return fixed if gold_l in fixed else ()


def _named_labels(cand: str, labels: tuple[str, ...]) -> set[str]:
    """Labels from the closed set that the candidate actually names."""
    text = " ".join(cand.lower().split())
    return {
        lab for lab in labels
        if re.search(rf"\b{re.escape(lab)}\b", text)
    }


def _score_label(datapoint: dict, cand: str, gold_s: str) -> float | None:
    """Single-committed-label scoring; ``None`` -> caller falls back to legacy."""
    labels = _closed_label_set(datapoint, gold_s)
    if not labels:
        return None
    named = _named_labels(cand, labels)
    if gold_s.strip().lower() not in named:
        return 0.0
    gold_list = set(_gold_labels(str(datapoint.get("answer", "")))) or {gold_s.lower()}
    return 1.0 if named <= gold_list else 0.0


def _score_user(cand: str, gold: Any) -> float | None:
    """Single-committed-user-id scoring; ``None`` -> caller falls back."""
    try:
        gold_id = int(str(gold).strip())
    except Exception:
        return None
    # "45,191" is one id, not two.
    ids = {int(n) for n in re.findall(r"\d+", re.sub(r"(?<=\d),(?=\d)", "", cand))}
    if len(ids) != 1:
        return 0.0
    return 1.0 if ids.pop() == gold_id else 0.0


def _synth_score(datapoint: dict, output: str) -> float:
    answer = str(datapoint.get("answer", ""))
    try:
        if "datetime" in answer:
            gold: Any = datetime.strptime(answer, "[datetime.date(%Y, %m, %d)]")
        else:
            gold = ast.literal_eval(answer)[0]
    except Exception:
        gold = answer

    gold_s = str(gold)
    cand = _extract_final_answer(output)
    if not cand:
        return 0.0

    # An exact (case-insensitive) match always scores.
    if cand == gold_s or cand.lower() == gold_s.lower():
        return 1.0

    atype = datapoint.get("answer_type", "")
    if atype == "ANSWER_TYPE.NUMERIC":
        # Exact integer match on the first number of the final answer; no partial credit.
        nums = re.findall(r"-?\d+", cand)
        try:
            return 1.0 if nums and int(nums[0]) == int(gold) else 0.0
        except Exception:
            return 0.0
    if atype == "ANSWER_TYPE.DATE":
        try:
            return 1.0 if dateutil.parser.parse(cand) == gold else 0.0
        except Exception:
            return 0.0

    # Comparison gold: the last comparison phrase in the final answer must be the gold phrase.
    if gold_s.lower() in [p.lower() for p in COMPARISON_PHRASES]:
        return 1.0 if _find_comparison_phrase(cand) == gold_s.lower() else 0.0

    if atype == "ANSWER_TYPE.LABEL":
        scored = _score_label(datapoint, cand, gold_s)
        if scored is not None:
            return scored
    elif atype == "ANSWER_TYPE.USER":
        scored = _score_user(cand, gold)
        if scored is not None:
            return scored

    # Fallback (other answer types, unknown label set, non-integer user gold): containment in the final answer.
    return 1.0 if gold_s.lower() in cand.lower() else 0.0


async def _score(info, state: vf.State, **_kw: Any) -> float:
    final = state.get("rlm_final_answer") or state.get("final_answer") or ""
    meta = json.loads(info) if isinstance(info, str) else info
    return _synth_score(meta, final)


_DEFAULT_MIN_CTX = 1024
_DEFAULT_MAX_CTX = 4096

# Task framing shown to the root model before the question, per task family.
_QUESTION_INSTRUCTIONS = {
    "trec_coarse": (
        "The context contains thousands of general-knowledge questions, one per "
        "line. Each line has a User ID and a question, and each question's answer "
        "falls into one of 6 categories: 'numeric value', 'entity', 'location', "
        "'description and abstract concept', 'abbreviation', 'human being'. "
        "Answer the following aggregate question."
    ),
    "spam": (
        "The context contains thousands of SMS text messages, one per "
        "line. Each line has a User ID and a text message, and each message's "
        "label is one of 2 categories: 'spam', 'ham' (i.e. not spam). "
        "Answer the following aggregate question."
    ),
}
_DEFAULT_QUESTION_INSTRUCTION = _QUESTION_INSTRUCTIONS["trec_coarse"]


def _build_dataset(
    *,
    dataset_name: str,
    min_ctx: int,
    max_ctx: int,
    num_examples: int,
    seed: int,
    exclude_numeric: bool,
) -> Dataset:
    def _keep(ex):
        if ex.get("dataset") != dataset_name:
            return False
        cl = ex.get("context_len", 0)
        if not (min_ctx <= cl <= max_ctx):
            return False
        if exclude_numeric and ex.get("answer_type") == "ANSWER_TYPE.NUMERIC":
            return False
        return True

    # with a cap take the first matches of the shuffled stream, else take all and shuffle
    if num_examples > 0:
        stream = load_dataset("oolongbench/oolong-synth", revision=OOLONG_SYNTH_REVISION, split="validation", streaming=True)
        if seed is not None:
            stream = stream.shuffle(seed=seed, buffer_size=10_000)
        samples: list[dict] = []
        for ex in stream:
            if _keep(ex):
                samples.append(ex)
                if len(samples) >= num_examples:
                    break
    else:
        ds = load_dataset("oolongbench/oolong-synth", revision=OOLONG_SYNTH_REVISION, split="validation")
        meta = ds.select_columns(["dataset", "context_len", "answer_type"])
        samples = [ds[i] for i, m in enumerate(meta) if _keep(m)]
        if seed is not None:
            random.Random(seed).shuffle(samples)

    instruction = _QUESTION_INSTRUCTIONS.get(dataset_name, _DEFAULT_QUESTION_INSTRUCTION)
    rows: list[dict] = []
    for i, s in enumerate(samples):
        question = s["question"]
        context = s.get("context_window_text", s.get("context", ""))
        meta = {
            "id": s.get("id"),
            "dataset": s.get("dataset", ""),
            "answer_type": s.get("answer_type", ""),
            "answer": str(s.get("answer", "")),
            "context": context,
            "root_prompt": f"{instruction}\n\nQuestion: {question}",
        }
        rows.append({
            "example_id": i,
            "prompt": [{"role": "user", "content": question}],
            "answer": str(s.get("answer", "")),
            "info": json.dumps(meta),
        })
    return Dataset.from_list(rows)


def load_environment(
    *,
    dataset_name: str = "trec_coarse",
    min_ctx: int = _DEFAULT_MIN_CTX,
    max_ctx: int = _DEFAULT_MAX_CTX,
    num_examples: int = -1,
    seed: int = 42,
    exclude_numeric: bool = False,
    max_iterations: int = 12,
    sub_max_tokens: int = 4096,
    min_iterations: int = 2,
    min_subcall: int = 1,
    max_sub_calls: int = 500,
    **kwargs: Any,
) -> vf.Environment:
    dataset = _build_dataset(
        dataset_name=dataset_name,
        min_ctx=min_ctx,
        max_ctx=max_ctx,
        num_examples=num_examples,
        seed=seed,
        exclude_numeric=exclude_numeric,
    )
    rubric = rlm_train.RLMTrainRubric(
        correctness=_score, weight=1.0, min_iterations=min_iterations,
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
