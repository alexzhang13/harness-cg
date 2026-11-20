"""Base Transformer (`singleturn-baseline`): the whole context in one prompt, one answer turn.

Each task reuses the backing env's dataset builder and grader:

    oolong                OOLONG-synth aggregation      (oolong-real)
    oolong_pairs          OOLONG-Pairs, pair-set F1     (oolong-pairs-real)
    graphwalks            GraphWalks, node-set F1       (graphwalks-real)
    longbenchpro          LongBench-Pro                 (longbenchpro-real)
    obliq                 OBLIQ-Bench ranking, nDCG     (obliq-bench-real; sub-task in ``obliq_task``)
    ada_leval_bestanswer  Ada-LEval BestAnswer          (ada-leval-real)
    mrcr                  MRCR v2                       (mrcr-v2-real)

Arguments: task, num_examples, seed, system_prompt, plus the backing env's dataset arguments. Harness-only
arguments such as max_iterations are ignored.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable
from typing import Any

import verifiers as vf
from datasets import Dataset

SUPPORTED_TASKS = (
    "oolong", "oolong_pairs", "graphwalks", "longbenchpro", "obliq", "ada_leval_bestanswer", "mrcr",
)


_DEFAULT_SYSTEM = (
    "You are a careful assistant. You are given a long document or problem "
    "followed by a question or task. Read all of it and answer directly in "
    "the exact format requested. Do not ask for tools or external help — "
    "produce the final answer yourself."
)

_TOKENIZER = None


def _as_dict(msg: Any) -> dict:
    """Message as a dict (at score time completions are pydantic objects)."""
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        try:
            return msg.model_dump()
        except Exception:
            pass
    return {
        k: getattr(msg, k, None)
        for k in ("role", "content", "reasoning_content", "thinking_blocks", "tokens", "completion_ids", "token_ids")
    }


def _msgs_text(completion: Any) -> str:
    """Answer text of a list or str completion."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for raw in reversed(completion):
            msg = _as_dict(raw)
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                t = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                if t:
                    return t
    return ""


def _decode_tokens(completion: Any) -> str:
    """Decode the completion token ids. With prime-rl's renderer, `content` is only filled after scoring."""
    if not isinstance(completion, list):
        return ""
    ids: list[int] = []
    for raw in completion:
        msg = _as_dict(raw)
        if msg.get("role") != "assistant":
            continue
        toks = msg.get("tokens") or {}
        if hasattr(toks, "model_dump"):
            try:
                toks = toks.model_dump()
            except Exception:
                toks = {}
        cand = (
            msg.get("completion_ids")
            or (toks.get("completion_ids") if isinstance(toks, dict) else None)
            or msg.get("token_ids")
        )
        if cand:
            ids = list(cand)
            break
    if not ids:
        return ""
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer

            model = os.environ.get(
                "SINGLETURN_TOKENIZER", "Qwen/Qwen3-30B-A3B-Instruct-2507"
            )
            _TOKENIZER = AutoTokenizer.from_pretrained(model)
        except Exception:
            _TOKENIZER = False
    if not _TOKENIZER:
        return ""
    try:
        return _TOKENIZER.decode(ids, skip_special_tokens=True)
    except Exception:
        return ""


def _completion_text(completion: Any, state: Any = None) -> str:
    """Assistant text at score time."""
    t = _msgs_text(completion)
    if t:
        return t
    if isinstance(state, dict):
        t = _msgs_text(state.get("completion"))
        if t:
            return t
    t = _decode_tokens(completion)
    if t:
        return t
    if isinstance(state, dict):
        t = _decode_tokens(state.get("completion"))
        if t:
            return t
    return ""


def _meta(info: Any) -> dict:
    if isinstance(info, str):
        try:
            return json.loads(info)
        except Exception:
            return {}
    return dict(info) if info else {}


def _build_singleturn_dataset(
    rows: list[dict],
    *,
    build_user: Callable[[dict], str],
    system_prompt: str,
) -> Dataset:
    """RLM-style rows -> single-turn rows with the context in the user prompt."""
    out: list[dict] = []
    for r in rows:
        meta = _meta(r.get("info"))
        out.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_user(meta)},
                ],
                "answer": r.get("answer", ""),
                "info": r.get("info") if r.get("info") is not None else {},
            }
        )
    return Dataset.from_list(out)


def _load_oolong(*, system_prompt: str, num_examples: int, seed: int, **a: Any):
    from oolong_real.env import _build_dataset, _synth_score  # type: ignore

    ds = _build_dataset(
        dataset_name=a.get("dataset_name", "trec_coarse"),
        min_ctx=int(a["min_ctx"]),
        max_ctx=int(a["max_ctx"]),
        num_examples=num_examples,
        seed=seed,
        exclude_numeric=bool(a.get("exclude_numeric", False)),
    )

    def build_user(meta: dict) -> str:
        return f"{meta.get('context', '')}\n\n{meta.get('root_prompt', '')}"

    data = _build_singleturn_dataset(list(ds), build_user=build_user, system_prompt=system_prompt)

    async def correctness(completion=None, info=None, **_kw):
        return float(_synth_score(_meta(info), _completion_text(completion, _kw.get("state"))))

    return data, vf.Rubric(funcs=[correctness], weights=[1.0])


def _load_graphwalks(*, system_prompt: str, num_examples: int, seed: int, **a: Any):
    from graphwalks_real.env import _build_dataset, _f1, _to_nodes  # type: ignore
    # skip_examples: skip the first N rows of the shuffle (rows disjoint from an N-row train stream)
    skip = int(a.get("skip_examples", 0))

    ds = _build_dataset(
        split=a.get("split", "128k_and_shorter"),
        num_examples=(num_examples + skip) if (skip and num_examples > 0) else num_examples,
        seed=seed,
        max_prompt_chars=a.get("max_prompt_chars"),
    )
    if skip:
        ds = ds.select(range(min(skip, len(ds)), len(ds)))

    fmt = (
        "\n\nWhen finished, end your reply with a single line exactly:\n"
        "Final Answer: [id1, id2, ...]\n"
        "a comma-separated list of node IDs (use [] for the empty set)."
    )

    def build_user(meta: dict) -> str:
        return f"{meta.get('context', '')}{fmt}"

    data = _build_singleturn_dataset(list(ds), build_user=build_user, system_prompt=system_prompt)

    async def correctness(completion=None, answer=None, **_kw):
        gold = json.loads(answer) if isinstance(answer, str) else answer
        return float(_f1(_to_nodes(_completion_text(completion, _kw.get("state"))), _to_nodes(gold)))

    return data, vf.Rubric(funcs=[correctness], weights=[1.0])


def _load_oolong_pairs(*, system_prompt: str, num_examples: int, seed: int, **a: Any):
    from oolong_pairs_real.env import (  # type: ignore
        SUPPORTED_CONTEXT_LENGTHS,
        _load_pairs,
        _load_trec_coarse_context,
        _score_f1,
    )

    ctx_len = int(a.get("context_length", 8192))
    if ctx_len not in SUPPORTED_CONTEXT_LENGTHS:
        raise ValueError(
            f"context_length={ctx_len!r} not bundled. Available: {SUPPORTED_CONTEXT_LENGTHS}"
        )

    questions = _load_pairs(ctx_len)
    rng = random.Random(seed if seed is not None else 0)
    rng.shuffle(questions)
    if num_examples and num_examples > 0:
        questions = questions[:num_examples]
    unlabeled, _labeled = _load_trec_coarse_context(ctx_len)

    fmt = (
        "\n\nList every matching pair of user IDs, one pair per line, each "
        "written as (id1, id2). If there are no matches, write: none."
    )

    rows: list[dict] = []
    for q in questions:
        gold = q.get("answer", []) or []
        rows.append(
            {
                "answer": json.dumps(list(gold)),
                "info": {
                    "context": unlabeled,
                    "raw_question": q["question"],
                },
            }
        )

    def build_user(meta: dict) -> str:
        return f"{meta.get('context', '')}\n\n{meta.get('raw_question', '')}{fmt}"

    data = _build_singleturn_dataset(rows, build_user=build_user, system_prompt=system_prompt)

    async def correctness(completion=None, answer=None, **_kw):
        gold = json.loads(answer) if isinstance(answer, str) else answer
        gold_list = gold if isinstance(gold, list) else [gold]
        return float(_score_f1(_completion_text(completion, _kw.get("state")), gold_list))

    return data, vf.Rubric(funcs=[correctness], weights=[1.0])


def _load_longbenchpro(*, system_prompt: str, num_examples: int, seed: int, **a: Any):
    from longbenchpro_real.env import _build_dataset, _compute  # type: ignore

    ds = _build_dataset(
        language=a.get("language", "English"),
        token_length=a.get("token_length", "32k"),
        thinking=bool(a.get("thinking", False)),
        num_examples=num_examples,
        seed=seed,
    )

    def build_user(meta: dict) -> str:
        return f"{meta.get('context', '')}\n\n{meta.get('root_prompt', '')}"

    data = _build_singleturn_dataset(list(ds), build_user=build_user, system_prompt=system_prompt)

    async def correctness(completion=None, info=None, **_kw):
        meta = _meta(info)
        return float(
            _compute(
                meta.get("secondary_task", ""),
                meta.get("answers", []),
                _completion_text(completion, _kw.get("state")),
            )
        )

    return data, vf.Rubric(funcs=[correctness], weights=[1.0])


def _singleturn_dataset_from_rows(
    rows: list[dict], *, build_user: Callable[[dict], str], system_prompt: str
) -> Dataset:
    """Like _build_singleturn_dataset, but build_user gets the whole row (obliq and mrcr need row["prompt"])."""
    out = [
        {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user(r)},
            ],
            "answer": r.get("answer", ""),
            "info": r.get("info") if r.get("info") is not None else {},
        }
        for r in rows
    ]
    return Dataset.from_list(out)


def _row_query(row: dict) -> str:
    p = row.get("prompt") or []
    return p[-1]["content"] if p else ""


def _load_obliq(*, system_prompt: str, num_examples: int, seed: int, **a: Any):
    """OBLIQ ranking with the candidate pool in the prompt; reward nDCG@50, other retrieval metrics logged."""
    from obliq_real.env import (  # type: ignore
        TASKS,
        _all_retrieval_metrics,
        _build_task_rows,
        _ndcg,
        _parse_metric,
        _parse_ranking,
        _recall,
    )

    subtask = a.get("obliq_task", "writing")
    rows = _build_task_rows(
        subtask,
        num_examples=num_examples,
        seed=seed,
        pool_size=int(a.get("pool_size", 300)),
        min_distractors=int(a.get("min_distractors", 10)),
        metric=a.get("metric", "ndcg@50"),
        doc_max_chars=int(a.get("doc_max_chars", 4000)),
        max_scan=a.get("max_scan", 250_000),
        start_id=0,
    )
    framing = TASKS[subtask]["framing"]

    def build_user(row: dict) -> str:
        meta = _meta(row["info"])
        docs = "\n\n".join(meta["context"])
        n = len(meta["context"])
        return (
            f"{framing}\n\nQuery:\n{_row_query(row)}\n\n"
            f"Below are {n} candidate documents, each formatted as "
            f"'--- doc_id: <id> --- <text>':\n\n{docs}\n\n"
            f"This is a RANKING task: the pool mixes a handful of truly relevant "
            f"documents with many distractors. Rank ALL {n} candidates by relevance "
            f"to the query, from MOST to LEAST relevant. Output ONLY a JSON array of "
            f"the bare doc_id strings (the text after 'doc_id:', no prefix), best "
            f'first, including every candidate — e.g. ["id_a", "id_b", ...]. Put ones '
            f"you are unsure about lower rather than dropping them."
        )

    data = _singleturn_dataset_from_rows(rows, build_user=build_user, system_prompt=system_prompt)

    def _rels(meta: dict) -> dict:
        return {str(c): int(s) for c, s in (meta.get("relevant") or {}).items()}

    async def correctness(completion=None, info=None, **_kw):
        meta = _meta(info)
        rels = _rels(meta)
        if not rels:
            return 0.0
        ranking = _parse_ranking(_completion_text(completion, _kw.get("state")))
        name, k = _parse_metric(str(meta.get("metric", "ndcg@50")))
        return float(_recall(ranking, rels, k) if name == "recall" else _ndcg(ranking, rels, k))

    rubric = vf.Rubric(funcs=[correctness], weights=[1.0])

    def _mk_metric(key: str):
        async def _metric(completion=None, info=None, **_kw):
            meta = _meta(info)
            ranking = _parse_ranking(_completion_text(completion, _kw.get("state")))
            return float(_all_retrieval_metrics(ranking, _rels(meta)).get(key, 0.0))
        _metric.__name__ = "obliq_" + key.replace("@", "")
        return _metric

    for key in ("ndcg@10", "recall@10", "recall@50", "recall@100"):
        rubric.add_reward_func(_mk_metric(key), weight=0.0)
    return data, rubric


def _load_ada_leval_bestanswer(*, system_prompt: str, num_examples: int, seed: int, **a: Any):
    """Ada-LEval BestAnswer with question and candidates in the prompt; exact match of the A<k> label."""
    from ada_leval_real.env import _build_ba_dataset, _coerce_label, _data_dir, _manifest_qids  # type: ignore

    include = (
        _manifest_qids(a.get("manifest_path"), a.get("manifest_split", "train"))
        if (a.get("manifest_path") or a.get("manifest_split")) else None
    )
    ds = _build_ba_dataset(
        setting=a.get("setting", "8k"),
        num_examples=num_examples,
        data_dir=_data_dir(a.get("data_dir")),
        include_qids=include,
    )

    def build_user(meta: dict) -> str:
        return (
            f"{meta.get('context', '')}\n\n"
            "Exactly one candidate is the correct/accepted answer; the rest are "
            "distractors. Answer with ONLY the label of the single best candidate, "
            "e.g. A17."
        )

    data = _build_singleturn_dataset(list(ds), build_user=build_user, system_prompt=system_prompt)

    async def correctness(completion=None, answer=None, **_kw):
        gold = _coerce_label(answer)
        pred = _coerce_label(_completion_text(completion, _kw.get("state")))
        return 1.0 if (pred is not None and gold is not None and pred == gold) else 0.0

    return data, vf.Rubric(funcs=[correctness], weights=[1.0])


def _load_mrcr(*, system_prompt: str, num_examples: int, seed: int, **a: Any):
    """MRCR v2 with the transcript in the prompt; same SequenceMatcher score as mrcr-v2-real."""
    from mrcr_v2_real.env import _PREAMBLE, _build_dataset, _mrcr_score  # type: ignore

    ds = _build_dataset(
        needle_count=int(a.get("needle_count", 2)),
        context_range=a.get("context_range", "32k-64k"),
        num_examples=num_examples,
        seed=seed,
    )

    def build_user(row: dict) -> str:
        meta = _meta(row["info"])
        question = _row_query(row)
        return (
            f"{_PREAMBLE}\n\n"
            f"--- BEGIN TRANSCRIPT ---\n{meta.get('context', '')}\n--- END TRANSCRIPT ---\n\n"
            f"{question}\n\n"
            f"Answer with the 12-character hash prefix followed by the reproduced "
            f"assistant response, EXACTLY."
        )

    data = _singleturn_dataset_from_rows(list(ds), build_user=build_user, system_prompt=system_prompt)

    async def correctness(completion=None, answer=None, **_kw):
        return float(_mrcr_score(_completion_text(completion, _kw.get("state")), answer or ""))

    return data, vf.Rubric(funcs=[correctness], weights=[1.0])


_LOADERS = {
    "oolong": _load_oolong,
    "mrcr": _load_mrcr,
    "oolong_pairs": _load_oolong_pairs,
    "graphwalks": _load_graphwalks,
    "longbenchpro": _load_longbenchpro,
    "obliq": _load_obliq,
    "ada_leval_bestanswer": _load_ada_leval_bestanswer,
}


def load_environment(
    *,
    task: str,
    num_examples: int = -1,
    seed: int = 42,
    system_prompt: str | None = None,
    **task_args: Any,
) -> vf.Environment:
    """Single-turn env for one task."""
    if task not in _LOADERS:
        raise ValueError(f"Unknown task {task!r}. Choose from {SUPPORTED_TASKS}.")

    sp = system_prompt if system_prompt is not None else _DEFAULT_SYSTEM
    loader = _LOADERS[task]
    dataset, rubric = loader(
        system_prompt=sp, num_examples=num_examples, seed=seed, **task_args
    )

    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)


__all__ = ["load_environment"]
