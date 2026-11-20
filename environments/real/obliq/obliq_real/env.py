"""OBLIQ-Bench retrieval (`obliq-real`): rank a pool of documents for a latent / implicit query.

Each query's pool is its gold positives plus random distractors, shuffled. The model writes a ranked list of doc ids;
reward is ``metric`` (nDCG@50 in the paper) against the qrels restricted to the pool.
"""

from __future__ import annotations

import ast
import json
import math
import os
import random
import re
import time
from functools import lru_cache
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

import rlm_train

REPO_ID = "dianetc/OBLIQ-Bench"
# OBLIQ-Bench v1.0 (the version used in the paper)
REVISION = os.environ.get("OBLIQ_BENCH_REVISION", "4ebee29f68ceeb62ca00bd73d5478acdd3cd7764")


def _hf_retry(fn, *, what: str, tries: int = 4):
    """Call ``fn`` with retries; HF Hub file resolution is occasionally flaky."""
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < tries - 1:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"HF load failed after {tries} tries: {what}") from last

TASKS: dict[str, dict[str, str]] = {
    "math": {
        "dir": "analogues/math",
        "framing": (
            "This is an analogical math-retrieval task. The query is a full "
            "mathematical problem. Find other problems in the candidate pool "
            "that require the SAME core reasoning pattern / proof strategy, "
            "even when the surface topic differs."
        ),
    },
    "writing": {
        "dir": "analogues/writing",
        "framing": (
            "This is an authorship/style-retrieval task. The query is a writing "
            "snippet. Find other snippets in the candidate pool written by the "
            "SAME author — matching voice and style across unrelated topics."
        ),
    },
    "twitter": {
        "dir": "descriptive/twitter",
        "framing": (
            "This is a descriptive stance-retrieval task. The query describes an "
            "IMPLICIT stance on a geopolitical conflict (irony, hedging, "
            "selective framing). Find tweets in the candidate pool that express "
            "that stance even without explicit keywords."
        ),
    },
    "wildchat": {
        "dir": "descriptive/wildchat",
        "framing": (
            "This is a descriptive behaviour-retrieval task. The query describes "
            "a latent failure mode in a Human–AI conversation. Find conversations "
            "in the candidate pool that exhibit that behaviour, visible in the "
            "transcript but never stated as a topic."
        ),
    },
    "congress": {
        "dir": "tip-of-tongue/congress",
        "framing": (
            "This is a tip-of-the-tongue retrieval task. The query is a hazy, "
            "first-person recollection of a congressional-hearing moment with no "
            "names, dates, or verbatim phrasing. Find the exact hearing passage "
            "in the candidate pool it refers to."
        ),
    },
}

ALL_TASKS = tuple(TASKS.keys())


@lru_cache(maxsize=len(TASKS))
def _load_qrels(task: str) -> dict[str, dict[str, int]]:
    """Return ``{query_id: {corpus_id: score}}``."""
    path = hf_hub_download(REPO_ID, f"{TASKS[task]['dir']}/queries+qrels/qrels.tsv", repo_type="dataset", revision=REVISION)
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, cid, score = parts[0].strip(), parts[1].strip(), parts[2].strip()
            try:
                s = int(float(score))
            except ValueError:
                continue
            if s <= 0:
                continue
            qrels.setdefault(qid, {})[cid] = s
    return qrels


@lru_cache(maxsize=len(TASKS))
def _load_queries(task: str) -> dict[str, str]:
    """Return ``{query_id: query_text}`` from the config's ``queries`` split."""
    ds = _hf_retry(lambda: load_dataset(REPO_ID, task, revision=REVISION, split="queries"),
                   what=f"queries/{task}")
    out: dict[str, str] = {}
    for row in ds:
        out[str(row["_id"])] = row.get("text") or ""
    return out


@lru_cache(maxsize=len(TASKS))
def _load_excluded(task: str) -> dict[str, set[str]]:
    """Return ``{query_id: {corpus_id, ...}}`` of docs to drop per query."""
    try:
        path = hf_hub_download(
            REPO_ID, f"{TASKS[task]['dir']}/queries+qrels/per_query_excluded_ids.json",
            repo_type="dataset", revision=REVISION,
        )
    except Exception:
        return {}
    raw = json.load(open(path, encoding="utf-8"))
    return {str(q): {str(c) for c in (ids or [])} for q, ids in raw.items()}


def _collect_corpus(
    task: str,
    needed_ids: frozenset[str],
    distractor_buffer: int,
    exclude_ids: frozenset[str],
    seed: int,
    max_scan: int | None,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Stream the corpus once; return (gold_texts, distractor_pool)."""
    ds = _hf_retry(lambda: load_dataset(REPO_ID, task, revision=REVISION, split="corpus", streaming=True),
                   what=f"corpus/{task}")
    needed = set(needed_ids)
    gold: dict[str, str] = {}
    reservoir: list[tuple[str, str]] = []
    rng = random.Random(seed)
    n_distractor_seen = 0
    scanned = 0
    for row in ds:
        scanned += 1
        rid = str(row["_id"])
        text = row.get("text") or ""
        is_gold = rid in needed
        if is_gold and rid not in gold:
            gold[rid] = text
        if not is_gold and rid not in exclude_ids:
            n_distractor_seen += 1
            if len(reservoir) < distractor_buffer:
                reservoir.append((rid, text))
            else:
                j = rng.randrange(n_distractor_seen)
                if j < distractor_buffer:
                    reservoir[j] = (rid, text)
        if len(gold) >= len(needed) and len(reservoir) >= distractor_buffer:
            break
        if max_scan is not None and scanned >= max_scan:
            break
    return gold, reservoir


def _render_doc(doc_id: str, text: str, max_chars: int) -> str:
    body = text if max_chars <= 0 else text[:max_chars]
    return f"---\ndoc_id: {doc_id}\n---\n{body}"


_DOCID_PREFIX_RE = re.compile(r"(?i)^doc[_\s]?id\s*[:=]\s*")


def _clean_id(x: Any) -> str:
    """Normalize one ranked entry to a bare doc_id (strip ``doc_id:`` labels, quotes, brackets, commas)."""
    _EDGE = "\"',[](){} \t\r\n"
    cid = str(x).strip().strip(_EDGE).strip()
    cid = _DOCID_PREFIX_RE.sub("", cid).strip().strip(_EDGE).strip()
    return cid


def _parse_ranking(answer: Any) -> list[str]:
    """Coerce the model's final answer into an ordered list of doc_id strings."""
    if answer is None:
        return []
    if isinstance(answer, (list, tuple)):
        items: Any = list(answer)
    else:
        s = str(answer).strip()
        if not s:
            return []
        items = None
        m = re.search(r"\[.*\]", s, flags=re.DOTALL)
        if m:
            blob = m.group(0)
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(blob)
                except Exception:
                    continue
                if isinstance(parsed, (list, tuple)):
                    items = list(parsed)
                    break
        if items is None:
            items = [re.sub(r"^[\s\-\*\d\.\)]+", "", ln).strip() for ln in s.splitlines()]
    out: list[str] = []
    seen: set[str] = set()
    for it in items:
        cid = _clean_id(it)
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _ndcg(ranking: list[str], rels: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, cid in enumerate(ranking[:k]):
        rel = rels.get(cid, 0)
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal) if r > 0)
    return dcg / idcg if idcg > 0 else 0.0


def _recall(ranking: list[str], rels: dict[str, int], k: int) -> float:
    if not rels:
        return 0.0
    topk = set(ranking[:k])
    hit = sum(1 for cid in rels if cid in topk)
    return hit / len(rels)


def _parse_metric(metric: str) -> tuple[str, int]:
    name, _, k = metric.lower().partition("@")
    return name.strip(), int(k) if k.strip().isdigit() else 10


def _all_retrieval_metrics(ranking: list[str], rels: dict[str, int]) -> dict[str, float]:
    if not rels:
        return {"ndcg@50": 0.0, "ndcg@10": 0.0, "recall@10": 0.0, "recall@50": 0.0, "recall@100": 0.0}
    return {
        "ndcg@50": _ndcg(ranking, rels, 50),
        "ndcg@10": _ndcg(ranking, rels, 10),
        "recall@10": _recall(ranking, rels, 10),
        "recall@50": _recall(ranking, rels, 50),
        "recall@100": _recall(ranking, rels, 100),
    }


def _compute_metrics(info: Any, state: vf.State) -> dict[str, float]:
    """All retrieval metrics for the rollout, computed once and cached on ``state``."""
    cached = state.get("_obliq_metrics")
    if cached is not None:
        return cached
    meta = json.loads(info) if isinstance(info, str) else (info or {})
    rels = {str(c): int(s) for c, s in (meta.get("relevant") or {}).items()}
    ranking = _parse_ranking(state.get("rlm_final_answer"))
    m = _all_retrieval_metrics(ranking, rels)
    state["_obliq_metrics"] = m
    return m


async def _score(info: Any, state: vf.State, **_kwargs: Any) -> float:
    """Reward: the metric named in ``meta["metric"]`` (default ndcg@50)."""
    meta = json.loads(info) if isinstance(info, str) else (info or {})
    rels = {str(c): int(s) for c, s in (meta.get("relevant") or {}).items()}
    if not rels:
        return 0.0
    m = _compute_metrics(info, state)
    name, k = _parse_metric(str(meta.get("metric", "ndcg@50")))
    key = f"{name}@{k}"
    if key in m:
        return m[key]
    ranking = _parse_ranking(state.get("rlm_final_answer"))
    return _recall(ranking, rels, k) if name == "recall" else _ndcg(ranking, rels, k)


async def obliq_ndcg10(info: Any, state: vf.State, **_kw: Any) -> float:
    return _compute_metrics(info, state)["ndcg@10"]


async def obliq_ndcg50(info: Any, state: vf.State, **_kw: Any) -> float:
    return _compute_metrics(info, state)["ndcg@50"]


async def obliq_recall10(info: Any, state: vf.State, **_kw: Any) -> float:
    return _compute_metrics(info, state)["recall@10"]


async def obliq_recall50(info: Any, state: vf.State, **_kw: Any) -> float:
    return _compute_metrics(info, state)["recall@50"]


async def obliq_recall100(info: Any, state: vf.State, **_kw: Any) -> float:
    return _compute_metrics(info, state)["recall@100"]


_AUX_METRICS = (obliq_ndcg50, obliq_ndcg10, obliq_recall10, obliq_recall50, obliq_recall100)


_ANSWER_PROTOCOL = (
    "This is a RANKING task: the candidate pool mixes a handful of truly "
    "relevant documents with many irrelevant distractors, and you are scored "
    "by where the relevant ones land in your ranking (nDCG@50 / recall@k). "
    "When you are done, inside a `repl` block set `answer[\"content\"]` to a "
    "list of candidate `doc_id` strings ordered from MOST to LEAST relevant "
    "(e.g. answer[\"content\"] = [\"id_a\", \"id_b\", ...]). Use the BARE id — "
    "the text after 'doc_id:' in each candidate, with NO 'doc_id:' prefix. "
    "Then set answer[\"ready\"] = True. Rank ALL candidates, best first — put the ones "
    "you are unsure about lower rather than dropping them, so borderline "
    "positives still count toward recall@50 and recall@100."
)


def _build_task_rows(
    task: str,
    num_examples: int,
    seed: int,
    pool_size: int,
    min_distractors: int,
    metric: str,
    doc_max_chars: int,
    max_scan: int | None,
    start_id: int,
    opaque_ids: bool = False,
) -> list[dict[str, Any]]:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {ALL_TASKS}")

    qrels = _load_qrels(task)
    queries = _load_queries(task)
    excluded = _load_excluded(task)

    qids = [q for q in qrels if q in queries]
    if not qids:
        raise RuntimeError(f"task={task!r}: no queries with both text and qrels")
    rng = random.Random(seed)
    rng.shuffle(qids)
    if num_examples > 0:
        # oversample: queries with gold docs beyond max_scan are skipped below
        qids = qids[: max(num_examples * 5, num_examples + 12)]

    # gold ids to fetch; excluded ids are removed from all pools
    needed: set[str] = set()
    exclude_all: set[str] = set()
    for qid in qids:
        needed.update(qrels[qid].keys())
        exclude_all.update(excluded.get(qid, set()))

    target = num_examples if num_examples > 0 else len(qids)
    distractor_buffer = max(min_distractors, pool_size) * 4 + 50
    gold_texts, distractor_pool = _collect_corpus(
        task,
        frozenset(needed),
        distractor_buffer=distractor_buffer,
        exclude_ids=frozenset(exclude_all),
        seed=seed,
        max_scan=max_scan,
    )

    rows: list[dict[str, Any]] = []
    framing = TASKS[task]["framing"]
    for qid in qids:
        if num_examples > 0 and len(rows) >= target:
            break
        excl = excluded.get(qid, set())
        positives = {
            cid: s for cid, s in qrels[qid].items()
            if cid in gold_texts and cid not in excl
        }
        if not positives:
            continue

        q_rng = random.Random(seed * 100003 + len(rows))
        n_distract = max(min_distractors, pool_size - len(positives))
        pos_ids = set(positives)
        avail = [(cid, txt) for cid, txt in distractor_pool if cid not in pos_ids and cid not in excl]
        q_rng.shuffle(avail)
        distractors = avail[:n_distract]

        pool: list[tuple[str, str]] = [(cid, gold_texts[cid]) for cid in positives]
        pool.extend(distractors)
        q_rng.shuffle(pool)
        if opaque_ids:
            # raw corpus ids can leak positives (sequential by source), so hash them per pool
            import hashlib as _h
            alias = {cid: "d" + _h.sha1(f"{seed}|{qid}|{cid}".encode()).hexdigest()[:10] for cid, _ in pool}
            pool = [(alias[cid], txt) for cid, txt in pool]
            positives = {alias[cid]: s for cid, s in positives.items()}

        context = [_render_doc(cid, txt, doc_max_chars) for cid, txt in pool]
        root_prompt = (
            f"{framing}\n\n"
            f"Query:\n{queries[qid]}\n\n"
            f"The REPL variable `context` is a list of {len(pool)} candidate documents, "
            f"each rendered over multiple lines as '---\\ndoc_id: <id>\\n---\\n<text>' "
            f"(so `e.split(chr(10))[1].split(': ', 1)[1]` is the id of entry `e`). Rank them by relevance to "
            f"the query above. {_ANSWER_PROTOCOL}"
        )
        meta = {
            "context": context,
            "root_prompt": root_prompt,
            "obliq_task": task,
            "query_id": qid,
            "relevant": positives,
            "pool_size": len(pool),
            "metric": metric,
        }
        rows.append(
            {
                "example_id": start_id + len(rows),
                "prompt": [{"role": "user", "content": queries[qid]}],
                "answer": json.dumps(sorted(positives)),
                "info": json.dumps(meta),
            }
        )
    if not rows:
        raise RuntimeError(
            f"task={task!r}: built 0 rows — no query had a retrievable gold doc "
            f"within max_scan={max_scan}. Raise max_scan or num_examples."
        )
    return rows


def load_environment(
    *,
    task: str = "math",
    num_examples: int = 20,
    seed: int = 42,
    pool_size: int = 50,
    min_distractors: int = 10,
    metric: str = "ndcg@50",
    doc_max_chars: int = 4000,
    max_scan: int | None = 250_000,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_iterations: int = 1,
    min_subcall: int = 0,
    opaque_ids: bool = False,
    **kwargs: Any,
) -> vf.Environment:
    """Load the OBLIQ-Bench real-sub-LLM retrieval env."""
    rows = _build_task_rows(
        task, num_examples=num_examples, seed=seed, pool_size=pool_size,
        min_distractors=min_distractors, metric=metric,
        doc_max_chars=doc_max_chars, max_scan=max_scan, start_id=0, opaque_ids=opaque_ids,
    )
    dataset = Dataset.from_list(rows)

    rubric = rlm_train.RLMTrainRubric(
        correctness=_score, weight=1.0,
        min_iterations=min_iterations, min_subcall=min_subcall,
    )
    for fn in _AUX_METRICS:
        rubric.add_metric(fn)
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = ["load_environment"]
