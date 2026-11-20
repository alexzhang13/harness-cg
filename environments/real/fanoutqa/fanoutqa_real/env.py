"""FanOutQA, evidence-provided setting (`fanoutqa-real`): multi-document Wikipedia QA.

`context` is the question's evidence articles from the local Wikipedia cache (see the package README). Metric
``"loose"``: fraction of gold strings found in the answer.
"""

from __future__ import annotations

import json
import random
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train

from .corpus import doc_text, require_wikicache
from .metrics import (
    METRICS,
    loose_accuracy,
    parse_final_answer,
    reference_strings,
)

SPLITS = ("dev",)
SETTINGS = ("evidence",)
CONTEXT_FORMATS = ("documents",)
_ROOT_PROMPT_TEMPLATE = """\
Answer this question about Wikipedia:

{question}

The REPL variable `context` holds the reference material: {shape_note} \
The question fans out over many of these articles - typically you must first \
read one article to enumerate the entities the question asks about, then look \
up one fact per entity in a different article. Articles are full Wikipedia \
pages in Markdown (tables and infoboxes included) and are far too long to read \
whole, so work programmatically: list the titles, search for the relevant \
strings, slice out the passages you need, and delegate per-article reading to \
sub-LM calls where that is cheaper than reading it yourself.

Submit a PLAIN TEXT final answer via `answer["content"] = ...; \
answer["ready"] = True`. If the question asks about several entities, give one \
line per entity in the form `Entity - value`, spelling out both the entity name \
and its value exactly as Wikipedia does (grading looks for each gold string \
inside your answer, so a bare list of values scores half marks). Do not submit a \
dict or a JSON blob - a plain string only.\
"""

_SHAPE_NOTES = {
    "documents": (
        "a list of {} dicts, each with a `title` and a `text` key ("
        "`len(context)` articles, `context[i]['title']`, `context[i]['text']`)."
    ),
}


def build_root_prompt(question: str, *, n_docs: int, context_format: str) -> str:
    return _ROOT_PROMPT_TEMPLATE.format(
        question=question.strip(),
        shape_note=_SHAPE_NOTES[context_format].format(n_docs),
    )


def score_answer(gold: Any, prediction: Any, *, metric: str = "loose") -> float:
    """Score one prediction against one gold answer (pure and deterministic)."""
    candidate = parse_final_answer(prediction)
    if not candidate.strip():
        return 0.0
    if metric == "loose":
        return loose_accuracy(gold, candidate)
    raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")


def _make_deterministic_score(metric: str):
    async def _score(answer: str, state: vf.State, **_kw: Any) -> float:
        gold = json.loads(answer) if isinstance(answer, str) else answer
        if gold is None:
            return 0.0
        final = state.get("rlm_final_answer")
        if final is None:
            final = state.get("final_answer")
        return score_answer(gold, final, metric=metric)

    return _score


def _questions(split: str) -> list[Any]:
    import fanoutqa

    return fanoutqa.load_dev()


def _unique_evidence(question: Any) -> list[Any]:
    """Deduplicated evidence articles for a question, in first-seen order."""
    seen: dict[int, Any] = {}
    for e in question.necessary_evidence:
        seen.setdefault(e.pageid, e)
    return list(seen.values())


def _count_subquestions(question: Any) -> int:
    n = 0
    stack = list(getattr(question, "decomposition", []) or [])
    while stack:
        sub = stack.pop()
        n += 1
        stack.extend(sub.decomposition)
    return n


class BuildStats:
    """Per-build drop accounting, attached to the env as ``env.fanoutqa_stats``."""

    def __init__(self) -> None:
        self.total = 0
        self.kept = 0
        self.dropped: dict[str, int] = {}
        self.missing_pageids: set[int] = set()

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    @property
    def drop_rate(self) -> float:
        return 0.0 if not self.total else 1.0 - self.kept / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped": dict(self.dropped),
            "drop_rate": round(self.drop_rate, 4),
            "missing_pageids": sorted(self.missing_pageids),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"BuildStats({self.to_dict()})"


def _build_dataset(
    *,
    split: str,
    setting: str,
    context_format: str,
    min_context_chars: int | None,
    max_context_chars: int | None,
    num_examples: int,
    seed: int,
    metric: str,
) -> tuple[Dataset, BuildStats]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}, got {setting!r}")
    if context_format not in CONTEXT_FORMATS:
        raise ValueError(f"context_format must be one of {CONTEXT_FORMATS}, got {context_format!r}")
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")

    cache = require_wikicache()
    questions = _questions(split)
    stats = BuildStats()
    stats.total = len(questions)

    order = list(range(len(questions)))
    random.Random(seed).shuffle(order)

    rows: list[dict] = []
    for idx in order:
        q = questions[idx]
        gold_ev = _unique_evidence(q)

        docs: list[dict[str, Any]] = []
        missing = False
        for e in gold_ev:
            text = doc_text(e, cache=cache)
            if text is None or not text.strip():
                stats.missing_pageids.add(e.pageid)
                missing = True
                break
            docs.append({"title": e.title, "url": e.url, "pageid": e.pageid, "text": text})
        if missing:
            stats.drop("evidence_not_in_cache")
            continue
        n_gold = len(docs)

        gold_titles = [e.title for e in gold_ev]
        distractor_titles = [d["title"] for d in docs if d["title"] not in set(gold_titles)]

        context: Any = [{"title": d["title"], "text": d["text"]} for d in docs]
        context_chars = sum(len(d["title"]) + len(d["text"]) for d in docs)

        if context_chars == 0:
            stats.drop("empty_context")
            continue
        if min_context_chars is not None and context_chars < min_context_chars:
            stats.drop("min_context_chars")
            continue
        if max_context_chars is not None and context_chars > max_context_chars:
            stats.drop("max_context_chars")
            continue

        gold = getattr(q, "answer", None)
        if not [r for r in reference_strings(gold) if str(r).strip()]:
            stats.drop("empty_gold")
            continue

        root_prompt = build_root_prompt(
            q.question, n_docs=len(docs), context_format=context_format
        )
        info = {
            "context": context,
            "root_prompt": root_prompt,
            "question": q.question,
            "fanoutqa_id": q.id,
            "fanoutqa_split": split,
            "fanoutqa_setting": setting,
            "fanoutqa_context_format": context_format,
            "fanoutqa_metric": metric,
            "fanoutqa_categories": list(q.categories),
            "fanoutqa_num_docs": len(docs),
            "fanoutqa_num_gold_docs": n_gold,
            "fanoutqa_num_distractor_docs": len(docs) - n_gold,
            "fanoutqa_gold_titles": gold_titles,
            "fanoutqa_distractor_titles": distractor_titles,
            "fanoutqa_context_chars": context_chars,
            "fanoutqa_num_subquestions": _count_subquestions(q),
            "fanoutqa_answer_type": type(gold).__name__ if gold is not None else "withheld",
        }
        rows.append(
            {
                "example_id": len(rows),
                "prompt": [{"role": "user", "content": q.question}],
                "answer": json.dumps(gold, ensure_ascii=False),
                "info": json.dumps(info, ensure_ascii=False),
            }
        )
        stats.kept += 1

        assert len(root_prompt) < 8_000, f"root_prompt too long: {len(root_prompt)}"
        assert context_chars > 0

        if num_examples and num_examples > 0 and len(rows) >= num_examples:
            break

    if not rows:
        raise ValueError(
            f"No fanoutqa rows for split={split!r} setting={setting!r} context_chars in "
            f"[{min_context_chars}, {max_context_chars}]. Drops: {stats.to_dict()}"
        )
    return Dataset.from_list(rows), stats


def load_environment(
    *,
    split: str = "dev",
    setting: str = "evidence",
    context_format: str = "documents",
    min_context_chars: int | None = None,
    max_context_chars: int | None = None,
    num_examples: int = -1,
    seed: int = 42,
    metric: str = "loose",
    max_iterations: int = 30,
    sub_max_tokens: int = 4096,
    min_subcall: int = 0,
    **kwargs: Any,
) -> vf.Environment:
    """Load FanOutQA (evidence-provided) wired through ``RLMTrainEnv``."""
    dataset, stats = _build_dataset(
        split=split,
        setting=setting,
        context_format=context_format,
        min_context_chars=min_context_chars,
        max_context_chars=max_context_chars,
        num_examples=num_examples,
        seed=seed,
        metric=metric,
    )

    correctness = _make_deterministic_score(metric)

    rubric = rlm_train.RLMTrainRubric(correctness=correctness, weight=1.0, min_subcall=min_subcall)
    env = rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )
    env.fanoutqa_stats = stats
    return env


__all__ = [
    "load_environment",
    "score_answer",
    "build_root_prompt",
    "BuildStats",
    "SPLITS",
    "SETTINGS",
    "CONTEXT_FORMATS",
    "METRICS",
]
