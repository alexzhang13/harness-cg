"""GraphWalks (`graphwalks-real`): BFS or parent queries over a directed-graph edge list.

The whole prompt is the ``context``; reward is set-F1 over node ids. Splits: ``128k_and_shorter``, ``256k_to_1mil``.
"""

from __future__ import annotations

import json
import re
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train

_REPO = "openai/graphwalks"
_SPLIT_FILES = {
    "128k_and_shorter": "graphwalks_128k_and_shorter.parquet",
    "256k_to_1mil": "graphwalks_256k_to_1mil.parquet",
}
_SEPARATOR = "Here is the graph to operate on"


def _to_nodes(answer: Any) -> list[str]:
    """Node ids from a list, or from the last 'Final Answer: [...]' / bare '[...]' line."""
    if isinstance(answer, (list, tuple)):
        return [str(x).strip().strip("'\"") for x in answer if str(x).strip()]
    if not isinstance(answer, str):
        answer = str(answer)
    text = answer.strip()
    m = None
    for line in reversed(text.splitlines()):
        m = re.search(r"final answer:\s*\[(.*?)\]\s*$", line, flags=re.IGNORECASE)
        if m:
            break
    if m is None:
        for line in reversed(text.splitlines()):
            m = re.search(r"^\s*\[(.*?)\]\s*$", line)
            if m:
                break
    if m is None:
        return []
    inner = m.group(1).strip()
    if not inner:
        return []
    return [tok.strip().strip("'\"") for tok in inner.split(",") if tok.strip()]


def _f1(pred: list[str], gold: list[str]) -> float:
    pred_set, gold_set = set(pred), set(gold)
    if not gold_set and not pred_set:
        return 1.0  # an empty gold set answered with an empty set
    overlap = len(pred_set & gold_set)
    recall = overlap / len(gold_set) if gold_set else 0.0
    precision = overlap / len(pred_set) if pred_set else 0.0
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


async def _score(answer: str, state: vf.State, **_kw: Any) -> float:
    gold = json.loads(answer) if isinstance(answer, str) else answer
    final = state.get("rlm_final_answer")
    if final is None:
        final = state.get("final_answer") or ""
    return _f1(_to_nodes(final), _to_nodes(gold))


_ROOT_PROMPT = (
    "The variable `context` holds a directed-graph problem: a worked example, then a large edge list "
    "(one `A -> B` per line), then the `Operation:` line you must actually perform.\n"
    "`Operation:` appears FIVE times in `context` -- four inside the worked example and once for real. "
    "The real one is the LAST occurrence, and the real edge list begins after the last "
    "`Here is the graph to operate on`; taking the first match instead solves the worked example.\n"
    "The operation is one of:\n"
    "- `Perform a BFS from node <X> and return only the nodes at exactly depth <D> (not nodes at "
    "intermediate depths).` Return the nodes whose shortest distance from X is EXACTLY D. Nodes at "
    "depth 1..D-1 are wrong answers, and so is X itself.\n"
    "- `Find the parents of node <X>.` Return every node with an edge pointing INTO X.\n"
    "Parse the edges into an adjacency structure in the REPL, execute the operation, and submit the "
    "resulting set of node IDs as your final answer (a Python list of node-id strings, e.g. "
    "`answer[\"content\"] = ['1679091c5a', 'c9f0f895fb']`; an empty set is `[]`). Scoring is set-F1, "
    "so order does not matter and extra nodes cost precision."
)


def _build_dataset(
    *,
    split: str,
    num_examples: int,
    seed: int | None,
    max_prompt_chars: int | None = None,
) -> Dataset:
    """Rows in seeded-shuffle order; ``max_prompt_chars`` drops longer prompts."""
    if split not in _SPLIT_FILES:
        raise ValueError(f"Unknown graphwalks split {split!r}. Choose from {tuple(_SPLIT_FILES)}")

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(_REPO, _SPLIT_FILES[split], repo_type="dataset")
    table = pq.read_table(path, columns=["prompt", "answer_nodes", "prompt_chars", "problem_type"])
    records = table.to_pylist()

    rng_order = list(range(len(records)))
    if seed is not None:
        import random
        random.Random(seed).shuffle(rng_order)

    rows: list[dict] = []
    for idx in rng_order:
        r = records[idx]
        pc = int(r["prompt_chars"])
        if max_prompt_chars is not None and pc > max_prompt_chars:
            continue
        prompt_text = r["prompt"]
        # the full prompt goes into `context`
        sep_idx = prompt_text.find(_SEPARATOR)
        question = prompt_text[:sep_idx].strip() if sep_idx != -1 else prompt_text[:2000]
        gold = list(r["answer_nodes"])
        info = {
            "context": prompt_text,
            "root_prompt": _ROOT_PROMPT,
            "problem_type": r["problem_type"],
            "prompt_chars": pc,
        }
        rows.append({
            "example_id": len(rows),
            "prompt": [{"role": "user", "content": question[:4000]}],
            "answer": json.dumps(gold),
            "info": json.dumps(info),
        })
        if num_examples and num_examples > 0 and len(rows) >= num_examples:
            break
    if not rows:
        raise ValueError(
            f"No graphwalks examples for split={split!r}, max_prompt_chars={max_prompt_chars}"
        )
    return Dataset.from_list(rows)


def load_environment(
    *,
    split: str = "128k_and_shorter",
    num_examples: int = -1,
    seed: int = 42,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_subcall: int = 0,
    **kwargs: Any,
) -> vf.Environment:
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


__all__ = ["load_environment"]
