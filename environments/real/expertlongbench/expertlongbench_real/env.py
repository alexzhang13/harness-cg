"""ExpertLongBench (`expertlongbench-real`): fill an expert checklist from a source document (T01, T03, T04, T06).

The model returns one JSON object mapping each checklist field to a value. Reward is the mean per-field equivalence
against the reference checklist; N/A golds need a matching decline, the rest go to one LLM judge call.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train
from rlm_train.judge_client import build_judge_client

HF_DATASET = "launch/ExpertLongBench"

# task -> (file in the HF repo, task line shown to the model)
TASKS: dict[str, tuple[str, str]] = {
    "T01LegalMDS": (
        "T01LegalMDS.jsonl",
        "You are a legal analyst. Read the court docket in `context` and "
        "produce a structured summary of the case.",
    ),
    "T03MaterialSEG": (
        "T03MaterialSEG.jsonl",
        "You are a materials scientist. Read the synthesis procedure in "
        "`context` and explain the rationale behind each experimental choice.",
    ),
    "T04EduPAE": (
        "T04EduPAE.jsonl",
        "You are an expert tutor. Read the tutoring dialogue in `context` and "
        "assess the student's most recent response.",
    ),
    "T06HealthCNG": (
        "T06HealthCNG.jsonl",
        "You are a clinician. Read the doctor-patient dialogue in `context` "
        "and produce the structured clinical note it supports.",
    ),
}

# Extra task spec for T03 (mostly N/A golds) and T04 (closed vocabularies).
TASK_SPEC: dict[str, str] = {
    "T03MaterialSEG": (
        "Fill a field only when the procedure states, or clearly implies, the reason for that choice; if it gives no "
        "reason, write N/A for that field."
    ),
    "T04EduPAE": (
        "Field values: \"Evaluation of the student response\" is one of \"Correct response\", \"Incorrect response\", "
        "\"Partially correct response\"; \"Mistake identification type\" and \"Mistake rectification type\" are each "
        "\"Direct\" or \"Indirect\" (N/A when there is no mistake); \"Repeated mistake\" is \"Yes\" or \"No\"."
    ),
}

# T04 labels are stored as letters
_T04_EVAL_FIELD = "Evaluation of the student response"
_T04_EVAL_CODES = {"a": "Incorrect response", "b": "Correct response", "c": "Partially correct response"}

_NA_VALUES = {"n/a", "na", "none", "not applicable", "not mentioned", ""}


def _is_na(v: Any) -> bool:
    return str(v if v is not None else "").strip().lower() in _NA_VALUES


def _stringify_input(raw: Any) -> str:
    """The ``input`` column as text."""
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, ensure_ascii=False, default=str)


def _parse_checklist(pred: str) -> dict[str, str]:
    """Pull a ``field -> value`` object out of the model's final answer."""
    txt = (pred or "").strip()
    for candidate in _json_candidates(txt):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return {str(k): ("" if v is None else str(v)) for k, v in obj.items()}
    out: dict[str, str] = {}
    for line in txt.splitlines():
        m = re.match(r"\s*[-*]?\s*([^:]{2,80}?)\s*:\s*(.+)$", line)
        if m:
            out[m.group(1).strip().strip('"')] = m.group(2).strip()
    return out


def _json_candidates(txt: str) -> list[str]:
    cands = []
    fence = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
    if fence:
        cands.append(fence.group(1).strip())
    first, last = txt.find("{"), txt.rfind("}")
    if first != -1 and last > first:
        cands.append(txt[first:last + 1])
    cands.append(txt)
    return cands


_JUDGE_SYSTEM = (
    "You grade expert-domain checklists. For each field you are given a "
    "reference value and a candidate value. Judge whether the candidate "
    "conveys the same substantive content as the reference. Ignore wording, "
    "ordering, and formatting differences. Be strict about facts, lenient "
    "about phrasing."
)

_JUDGE_TEMPLATE = """Task: {task}

Grade each field below.

{fields}

Return ONLY a JSON object mapping each field name to 1 (equivalent) or 0
(not equivalent). Include every field name exactly as given, and nothing else."""


def _make_score(judge_client: Any, judge_model: str, max_value_chars: int):
    async def _score(info, state: vf.State, **_kw: Any) -> float:
        meta = json.loads(info) if isinstance(info, str) else info
        gold: dict[str, Any] = meta.get("gold_checklist") or {}
        if not gold:
            return 0.0
        final = state.get("rlm_final_answer") or state.get("final_answer") or ""
        pred_text = final if isinstance(final, str) else str(final)
        if not pred_text.strip():
            return 0.0
        pred = _parse_checklist(pred_text)
        state["elb_parsed_fields"] = len(pred)

        pred_ci = {k.strip().lower(): v for k, v in pred.items()}

        scores: dict[str, float] = {}
        to_judge: list[tuple[str, str, str]] = []
        for field, gold_val in gold.items():
            cand = pred_ci.get(field.strip().lower(), "")
            if _is_na(gold_val):   # N/A gold: credit only a matching decline
                scores[field] = 1.0 if _is_na(cand) else 0.0
                continue
            if _is_na(cand):
                scores[field] = 0.0
                continue
            if str(gold_val).strip().lower() == str(cand).strip().lower():
                scores[field] = 1.0
                continue
            to_judge.append((field, str(gold_val), str(cand)))

        if to_judge:
            if judge_client is None:
                state["elb_judge_skipped"] = True
                for field, _, _ in to_judge:
                    scores[field] = 0.0
            else:
                verdicts = await _judge_fields(
                    judge_client, judge_model, meta.get("task_description") or "",
                    to_judge, max_value_chars, state,
                )
                for field, _, _ in to_judge:
                    scores[field] = float(verdicts.get(field, 0.0))

        state["elb_field_scores"] = scores
        return sum(scores.values()) / len(scores) if scores else 0.0

    return _score


async def _judge_fields(
    client: Any, model: str, task_desc: str,
    items: list[tuple[str, str, str]], max_value_chars: int, state: vf.State,
) -> dict[str, float]:
    blocks = []
    for field, gold_val, cand in items:
        blocks.append(
            f"### Field: {field}\n"
            f"Reference: {gold_val[:max_value_chars]}\n"
            f"Candidate: {cand[:max_value_chars]}"
        )
    prompt = _JUDGE_TEMPLATE.format(task=task_desc, fields="\n\n".join(blocks))
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max(256, 24 * len(items)),
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        state["elb_judge_error"] = repr(e)[:200]
        return {}
    if not text:   # judge failure, not a zero score
        state["elb_judge_empty"] = True
        return {}
    for candidate in _json_candidates(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out: dict[str, float] = {}
            lower = {str(k).strip().lower(): v for k, v in obj.items()}
            for field, _, _ in items:
                v = lower.get(field.strip().lower())
                try:
                    out[field] = 1.0 if float(v) >= 0.5 else 0.0
                except (TypeError, ValueError):
                    out[field] = 1.0 if str(v).strip().lower() in ("yes", "true") else 0.0
            return out
    state["elb_judge_unparsed"] = text[:200]
    return {}


def _build_dataset(
    *, tasks: list[str], num_examples: int, seed: int | None,
    max_context_chars: int | None,
) -> Dataset:
    from huggingface_hub import hf_hub_download

    raw: list[tuple[str, dict]] = []
    for task in tasks:
        fn, _desc = TASKS[task]
        path = hf_hub_download(HF_DATASET, fn, repo_type="dataset")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    raw.append((task, json.loads(line)))

    if seed is not None:
        import random

        random.Random(seed).shuffle(raw)
    if num_examples and num_examples > 0:
        raw = raw[:num_examples]

    rows: list[dict] = []
    for i, (task, r) in enumerate(raw):
        checklist = r.get("human_reference_checklist")
        if not isinstance(checklist, dict) or not checklist:
            continue
        ctx = _stringify_input(r.get("input"))
        if not ctx.strip():
            continue
        if max_context_chars is not None:
            ctx = ctx[:max_context_chars]

        desc = TASKS[task][1]
        field_list = "\n".join(f"  - {k}" for k in checklist)
        gold = {str(k): ("" if v is None else str(v)) for k, v in checklist.items()}
        spec = ""
        if task == "T04EduPAE" and gold.get(_T04_EVAL_FIELD, "").strip().lower() in _T04_EVAL_CODES:
            gold[_T04_EVAL_FIELD] = _T04_EVAL_CODES[gold[_T04_EVAL_FIELD].strip().lower()]
        if TASK_SPEC.get(task):
            spec = f" {TASK_SPEC[task]}"
        info = {
            "context": ctx,
            "root_prompt": (
                f"{desc}\n\n"
                "Fill in the following expert checklist. Every field must be "
                "present.\n\n"
                f"{field_list}\n\n"
                "Write 'N/A' for a field the source genuinely does not "
                f"support — do not invent content.{spec} Submit a single JSON object "
                "mapping each field name above to its value, and nothing else."
            ),
            "gold_checklist": gold,
            "task_description": desc,
            "elb_task": task,
            "item_id": r.get("id") or "",
        }
        rows.append({
            "example_id": i,
            "prompt": [{"role": "user", "content": "Fill in the expert checklist."}],
            "answer": json.dumps(checklist, ensure_ascii=False)[:2000],
            "info": json.dumps(info, ensure_ascii=False),
        })
    return Dataset.from_list(rows)


def load_environment(
    *,
    task: str | Sequence[str] | None = None,
    num_examples: int = -1,
    seed: int = 42,
    max_context_chars: int | None = None,
    max_value_chars: int = 2000,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_subcall: int = 1,
    judge_client: Any | None = None,
    judge_model: str = "openai/gpt-4.1-mini",
    **kwargs: Any,
) -> vf.Environment:
    """Load ExpertLongBench as an RLMTrainEnv. task: a ``TASKS`` name, a list of them, or None for all.

    judge_model must be a non-reasoning model (the verdict is read from the reply content).
    """
    if task is None:
        tasks = list(TASKS)
    else:
        tasks = [task] if isinstance(task, str) else list(task)
        unknown = [x for x in tasks if x not in TASKS]
        if unknown:
            raise ValueError(
                f"task={unknown!r} unknown. Choose from {tuple(TASKS)}."
            )
        if not tasks:
            raise ValueError("task=[] selects nothing; pass None for all tasks.")

    dataset = _build_dataset(
        tasks=tasks, num_examples=num_examples, seed=seed,
        max_context_chars=max_context_chars,
    )

    if judge_client is not None:
        client, model = judge_client, judge_model
    elif (
        os.environ.get("RLM_JUDGE_API_KEY")
        or os.environ.get("RLM_JUDGE_BASE_URL")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("PRIME_API_KEY")
    ):
        client, model = build_judge_client(model=judge_model)
    else:
        raise RuntimeError(
            "expertlongbench-real needs RLM_JUDGE_API_KEY / RLM_JUDGE_BASE_URL, "
            "OPENAI_API_KEY or PRIME_API_KEY for the checklist judge. Pass "
            "`judge_client` explicitly to override."
        )

    rubric = rlm_train.RLMTrainRubric(
        correctness=_make_score(client, model, max_value_chars),
        weight=1.0, min_subcall=min_subcall,
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = ["load_environment", "TASKS"]
