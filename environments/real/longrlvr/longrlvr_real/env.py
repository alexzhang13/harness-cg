"""LongRLVR (`longrlvr-real`): long-document QA where the model also cites the chunks it used.

Reward (upstream ``longrl_reward_manager``): ``(ans + 0.1 * fbeta + 0.9 * ans * fbeta) / 2``, with ``ans`` from an
LLM judge and ``fbeta`` the F2 between cited and reference chunk ids.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Iterator, Sequence
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train
from rlm_train.judge_client import build_judge_client

_REPO = "Guanzheng/LongRLVR-Data"

_FILES: dict[tuple[str, str], str] = {
    ("llama", "train"): "llama/train_all_data_64k_final.parquet",
    ("llama", "test"): "llama/test_all_data_64k_final.parquet",
    ("qwen2.5-14b", "train"): "qwen2.5-14b/train_data_64k_final.parquet",
    ("qwen2.5-14b", "test"): "qwen2.5-14b/test_data_64k_final.parquet",
    ("qwen2.5-7b", "train"): "qwen2.5-7b/train_data_64k.parquet",
    ("qwen2.5-7b", "test"): "qwen2.5-7b/test_data_64k_128k.parquet",
}
SUBSETS = ("llama", "qwen2.5-14b", "qwen2.5-7b")
SPLITS = ("train", "test")

CHARS_PER_TOKEN = 3.5

_TOKEN_BUCKETS: dict[str, tuple[int, int | None]] = {
    "8k": (0, 8 * 1024),
    "16k": (8 * 1024, 16 * 1024),
    "32k": (16 * 1024, 32 * 1024),
    "64k": (32 * 1024, 64 * 1024),
    "128k": (64 * 1024, 128 * 1024),
    "256k": (128 * 1024, 256 * 1024),
    "any": (0, None),
}
LENGTH_BUCKETS = tuple(_TOKEN_BUCKETS)


_HEADER_PREFIX = (
    "You are given a long document split into labeled chunks (<CHUNK_0>...</CHUNK_0>"
)
_DOC_SEP = "\n\nDocument:\n"
_Q_SEP = "\n\nQuestion: "
_CHUNK_RE = re.compile(r"<CHUNK_(\d+)>\n(.*?)\n</CHUNK_(\d+)>", re.DOTALL)

_MIN_BODY_COVERAGE = 0.85

MAX_ROOT_PROMPT_CHARS = 3000
_MAX_QUESTION_CHARS = 2000


class ParseError(Exception):
    """Raised when an upstream row does not match the expected layout."""


def parse_upstream_prompt(content: str) -> tuple[list[str], str]:
    """Split one upstream prompt string into ``(chunk_bodies, question)``."""
    if not isinstance(content, str) or not content.startswith(_HEADER_PREFIX):
        raise ParseError("missing LongRLVR instruction header")
    doc_i = content.find(_DOC_SEP)
    if doc_i < 0:
        raise ParseError("missing 'Document:' separator")
    q_i = content.rfind(_Q_SEP)
    if q_i <= doc_i:
        raise ParseError("missing trailing 'Question:' separator")

    doc = content[doc_i + len(_DOC_SEP) : q_i]
    question = content[q_i + len(_Q_SEP) :].strip()
    if not question:
        raise ParseError("empty question")

    matches = list(_CHUNK_RE.finditer(doc))
    if not matches:
        raise ParseError("no <CHUNK_i> blocks found")
    bodies: list[str] = []
    for expected, m in enumerate(matches):
        if m.group(1) != m.group(3):
            raise ParseError(f"unbalanced chunk tags at index {expected}")
        if int(m.group(1)) != expected:
            raise ParseError(
                f"non-contiguous chunk ids: expected {expected}, got {m.group(1)}"
            )
        bodies.append(m.group(2))
    coverage = sum(len(b) for b in bodies) / max(1, len(doc))
    if coverage < _MIN_BODY_COVERAGE:
        raise ParseError(f"chunk bodies cover only {coverage:.1%} of the document")
    return bodies, question


_TAG_CHUNKS_RE = re.compile(r"<useful_?chunks>(.*?)</useful_?chunks>", re.DOTALL | re.I)
_TAG_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.I)
_CHUNK_REF_RE = re.compile(r"<?\s*CHUNK[_ ]?(\d+)\s*>?", re.I)
_LOOSE_CHUNKS_RE = re.compile(
    r"useful[_ ]?chunks['\"]?\s*[:=]\s*\[([^\]]*)\]?", re.I
)

_ANSWER_KEYS = ("answer", "content", "final_answer", "response", "text")
_CHUNK_KEYS = ("useful_chunks", "useful chunks", "chunks", "used_chunks", "ref_chunks", "evidence")


def _coerce_chunk_ids(value: Any) -> list[int]:
    """Coerce a model-supplied chunk reference into a list of ints."""
    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(x) for x in _CHUNK_REF_RE.findall(value)] or [
            int(x) for x in re.findall(r"\d+", value)
        ]
    if isinstance(value, (list, tuple, set)):
        out: list[int] = []
        for item in value:
            out.extend(_coerce_chunk_ids(item))
        return out
    return []


def parse_final_answer(final: Any) -> tuple[str, list[int]]:
    """Coerce ``state["rlm_final_answer"]`` into ``(answer_text, chunk_ids)``."""
    if final is None:
        return "", []

    if isinstance(final, dict):
        answer = ""
        for key in _ANSWER_KEYS:
            if key in final and final[key] is not None:
                answer = final[key]
                break
        else:
            answer = {k: v for k, v in final.items() if k not in _CHUNK_KEYS}
        chunks: list[int] = []
        for key in _CHUNK_KEYS:
            if key in final:
                chunks = _coerce_chunk_ids(final[key])
                break
        if not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False, default=str)
        return answer.strip(), chunks

    if isinstance(final, (list, tuple)):
        return json.dumps(list(final), ensure_ascii=False, default=str), []

    text = final if isinstance(final, str) else str(final)
    stripped = text.strip()
    if stripped.startswith("{"):
        import ast

        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(stripped)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict):
                return parse_final_answer(obj)

    m_chunks = _TAG_CHUNKS_RE.search(text)
    m_answer = _TAG_ANSWER_RE.search(text)
    if m_chunks or m_answer:
        answer = (m_answer.group(1) if m_answer else text).strip()
        chunks = _coerce_chunk_ids(m_chunks.group(1)) if m_chunks else []
        return answer, chunks

    m = _LOOSE_CHUNKS_RE.search(text)
    if m:
        return stripped, _coerce_chunk_ids(m.group(1))
    return stripped, []


def context_scores(used: Sequence[int], ref: Sequence[int]) -> dict[str, float]:
    """Upstream ``longrl_reward_manager`` chunk scoring, verbatim."""
    used = list(used)
    ref_set = set(ref)
    if not used or not ref_set:
        return {"precision": 0.0, "recall": 0.0, "fbeta": 0.0}
    inter = len(set(used) & ref_set)
    precision = inter / len(used)
    recall = inter / len(ref_set)
    if precision + recall == 0:
        fbeta = 0.0
    else:
        fbeta = (5 * precision * recall) / (4 * precision + recall)
    return {"precision": precision, "recall": recall, "fbeta": fbeta}


ANSWER_METRICS = ("judge",)

_JUDGE_SYSTEM_MSG = (
    "You are an expert QA evaluator. Given a question, reference context (document chunks), a reference answer, "
    "and a generated answer, judge whether the generated answer expresses the same core meaning as the reference "
    "answer with respect to the question. Use the reference context only to clarify facts and scope; focus on "
    "semantic equivalence, not wording. Respond with exactly one token: [[Yes]] or [[No]]. No explanations."
)

_JUDGE_CONTEXT_CAP = 12000


def _judge_user_msg(question: str, context_text: str, ground_truth: str, answer: str) -> str:
    """Upstream ``evaluate_answer`` user message, verbatim (incl. whitespace)."""
    return f"""
Question:
{question}

Reference Context:
<REF_CONTEXT>
{context_text}
</REF_CONTEXT>

Reference Answer:
{ground_truth}

Generated Answer:
{answer}

Judgment rules:
1) Same facts and intent relative to the question → [[Yes]].
2) Different facts, changed scope, or missing key information → [[No]].
3) Partial overlap is not enough; omissions count as different.
4) Grammar/wording/order do not matter—only meaning.

Your direct response (choose one): [[Yes]] / [[No]]
"""


_BRACKET_JUDGMENT_RE = re.compile(r"\[\[(YES|NO)\]\]")


def parse_judgment(response_text: str) -> float | None:
    """Upstream ``extract_binary_judgment_from_response``."""
    if not isinstance(response_text, str):
        return None
    cleaned = response_text.strip().upper()
    m = _BRACKET_JUDGMENT_RE.search(cleaned)
    if m:
        return 1.0 if m.group(1) == "YES" else 0.0
    if "YES" in cleaned:
        return 1.0
    if "NO" in cleaned:
        return 0.0
    return None


def _make_judge_client(judge_model: str) -> tuple[Any, str]:
    """Resolve the answer-judge endpoint."""
    if not (
        os.environ.get("RLM_JUDGE_API_KEY")
        or os.environ.get("RLM_JUDGE_BASE_URL")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("PRIME_API_KEY")
    ):
        raise RuntimeError(
            "longrlvr-real with answer_metric='judge' needs RLM_JUDGE_API_KEY/"
            "RLM_JUDGE_BASE_URL, OPENAI_API_KEY, or PRIME_API_KEY."
        )
    return build_judge_client(default_model=judge_model)


async def _judge_answer(
    client: Any,
    model: str,
    *,
    question: str,
    gold: str,
    pred: str,
    ref_chunks_text: Sequence[str] | str,
    state: vf.State,
) -> float:
    if not pred.strip():
        return 0.0
    context_text = (
        ref_chunks_text if isinstance(ref_chunks_text, str) else "\n".join(ref_chunks_text or [])
    )
    if len(context_text) > _JUDGE_CONTEXT_CAP:
        context_text = context_text[:_JUDGE_CONTEXT_CAP]
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_MSG},
        {"role": "user", "content": _judge_user_msg(question, context_text, gold, pred)},
    ]
    for kwargs in ({"temperature": 0.0}, {}):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=512,
                **kwargs,
            )
            text = (resp.choices[0].message.content or "").strip()
            verdict = parse_judgment(text)
            if verdict is None:
                state["longrlvr_judge_unparsed"] = text[:200]
                return 0.0
            return verdict
        except Exception as e:  # noqa: BLE001  retry once without temperature
            state["longrlvr_judge_error"] = repr(e)[:200]
    return 0.0


_UPSTREAM_MAX = 2.0


def combine(answer_acc: float, fbeta: float, *, normalize: bool = True) -> float:
    """Upstream synergistic combination::"""
    raw = answer_acc + 0.1 * fbeta + 0.9 * answer_acc * fbeta
    return raw / _UPSTREAM_MAX if normalize else raw


def _decode(blob: Any) -> Any:
    return json.loads(blob) if isinstance(blob, str) else blob


def make_reward(*, answer_metric: str = "judge", judge_model: str = "openai/gpt-4.1-mini"):
    """Build the LongRLVR correctness function (and its metric companions)."""
    if answer_metric not in ANSWER_METRICS:
        raise ValueError(f"answer_metric must be one of {ANSWER_METRICS}, got {answer_metric!r}")

    client, model = _make_judge_client(judge_model)

    async def _components(answer: Any, info: Any, state: vf.State) -> dict[str, float]:
        gold = _decode(answer) or {}
        meta = _decode(info) or {}
        gold_text = gold.get("answer", "") if isinstance(gold, dict) else str(gold)
        ref_chunks = gold.get("useful_chunks", []) if isinstance(gold, dict) else []
        question = meta.get("question") or ""

        final = state.get("rlm_final_answer")
        if final is None:
            final = state.get("final_answer")
        pred_text, pred_chunks = parse_final_answer(final)

        ctx = context_scores(pred_chunks, ref_chunks)
        format_ok = 1.0 if (pred_text and pred_chunks) else 0.0

        answer_acc = await _judge_answer(
            client, model,
            question=question, gold=gold_text, pred=pred_text,
            ref_chunks_text=meta.get("ref_chunks_text") or [],
            state=state,
        )

        comp = {
            "answer_acc": answer_acc,
            "precision": ctx["precision"],
            "recall": ctx["recall"],
            "fbeta": ctx["fbeta"],
            "format_ok": format_ok,
            "n_pred_chunks": float(len(pred_chunks)),
            "raw_score": combine(answer_acc, ctx["fbeta"], normalize=False),
        }
        state["longrlvr"] = comp
        return comp

    async def longrlvr_reward(answer: Any = "", info: Any = "", state: vf.State = None, **_kw: Any) -> float:
        comp = await _components(answer, info, state if state is not None else {})
        return combine(comp["answer_acc"], comp["fbeta"], normalize=True)

    def _metric(name: str, key: str):
        async def fn(state: vf.State = None, **_kw: Any) -> float:
            comp = (state or {}).get("longrlvr") or {}
            return float(comp.get(key, 0.0))

        fn.__name__ = name
        return fn

    metrics = [
        _metric("longrlvr_answer_acc", "answer_acc"),
        _metric("longrlvr_context_fbeta", "fbeta"),
        _metric("longrlvr_context_precision", "precision"),
        _metric("longrlvr_context_recall", "recall"),
        _metric("longrlvr_format_ok", "format_ok"),
        _metric("longrlvr_n_pred_chunks", "n_pred_chunks"),
        _metric("longrlvr_raw_score", "raw_score"),
    ]
    return longrlvr_reward, metrics


_ANSWER_PROTOCOL = (
    "Submit a dict with exactly two keys:\n"
    '  answer["content"] = {"answer": "<self-contained answer text>", '
    '"useful_chunks": [<chunk ids>]}\n'
    '  answer["ready"] = True\n'
    "`useful_chunks` must be the integer ids of ONLY the chunks whose text "
    "actually supports your answer. Both precision and recall are scored "
    "(recall counts double), so do not pad the list with maybes and do not "
    "leave out a chunk you relied on. Answer in a full self-contained "
    "sentence — it is graded against a reference answer for meaning, not "
    "wording."
)


def _build_root_prompt(question: str, n_chunks: int) -> str:
    shape = (
        f"The REPL variable `context` is a dict with keys `question` (str) and "
        f"`chunks` (list of {n_chunks} strings; `context['chunks'][i]` is the "
        f"body of chunk i, i.e. chunk ids run 0..{n_chunks - 1}). The chunks are "
        f"consecutive slices of one long document."
    )
    return (
        f"{shape}\n\n"
        f"Question: {question}\n\n"
        f"Locate the chunks that contain the evidence for this question, read them, "
        f"and answer from them. {_ANSWER_PROTOCOL}"
    )


def _iter_records(path: str, seed: int | None) -> Iterator[dict]:
    """Stream rows one parquet row-group at a time."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    order = list(range(pf.metadata.num_row_groups))
    rng = random.Random(seed) if seed is not None else None
    if rng is not None:
        rng.shuffle(order)
    for rg in order:
        rows = pf.read_row_group(rg, columns=["prompt", "reward_model", "extra_info"]).to_pylist()
        if rng is not None:
            rng.shuffle(rows)
        yield from rows
        del rows


def _resolve_length(length: str | Sequence[str] | None) -> list[tuple[int, int | None]]:
    if length is None or length == "any":
        buckets: list[tuple[int, int | None]] = []
    else:
        names = [length] if isinstance(length, str) else list(length)
        bad = [n for n in names if n not in _TOKEN_BUCKETS]
        if bad:
            raise ValueError(f"unknown length bucket(s) {bad}; choose from {LENGTH_BUCKETS}")
        buckets = []
        for n in names:
            lo_tok, hi_tok = _TOKEN_BUCKETS[n]
            lo = int(lo_tok * CHARS_PER_TOKEN)
            hi = None if hi_tok is None else int(hi_tok * CHARS_PER_TOKEN)
            buckets.append((lo, hi))
    return buckets


def _length_ok(n_chars: int, buckets: list[tuple[int, int | None]]) -> bool:
    if not buckets:
        return True
    for lo, hi in buckets:
        if n_chars >= lo and (hi is None or n_chars < hi):
            return True
    return False


def _build_dataset(
    *,
    subsets: Sequence[str],
    split: str,
    length: str | Sequence[str] | None,
    num_examples: int,
    seed: int | None,
    stats: dict[str, Any] | None = None,
) -> Dataset:
    from huggingface_hub import hf_hub_download

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    buckets = _resolve_length(length)

    rows: list[dict] = []
    seen_questions: set[str] = set()
    drops: dict[str, int] = {}
    scanned = 0

    def _drop(reason: str) -> None:
        drops[reason] = drops.get(reason, 0) + 1

    def _process(rec: dict, subset: str) -> dict | None:
        content = (rec.get("prompt") or [{}])[0].get("content")
        if not isinstance(content, str):
            _drop("no_prompt_content")
            return None
        if not _length_ok(len(content), buckets):
            _drop("length_filter")
            return None
        extra = rec.get("extra_info") or {}
        ref_chunks = list(extra.get("ref_chunks") or [])
        if len(ref_chunks) < 1:
            _drop("too_few_ref_chunks")
            return None
        gold = (rec.get("reward_model") or {}).get("ground_truth") or ""
        if not str(gold).strip():
            _drop("empty_ground_truth")
            return None
        try:
            chunk_bodies, tail_question = parse_upstream_prompt(content)
        except ParseError as e:
            _drop("parse:" + str(e).split(":", 1)[0])
            return None
        if not set(ref_chunks) <= set(range(len(chunk_bodies))):
            _drop("ref_chunk_out_of_range")
            return None
        question = (extra.get("question") or tail_question).strip()
        if len(question) > _MAX_QUESTION_CHARS:
            _drop("question_too_long")
            return None
        if question in seen_questions:
            _drop("duplicate_question")
            return None
        seen_questions.add(question)

        context = {"question": question, "chunks": chunk_bodies}

        root_prompt = _build_root_prompt(question, len(chunk_bodies))
        if len(root_prompt) > MAX_ROOT_PROMPT_CHARS:  # pragma: no cover - guarded above
            _drop("root_prompt_too_long")
            return None

        info = {
            "context": context,
            "root_prompt": root_prompt,
            "question": question,
            "gold_answer": str(gold),
            "ref_chunks": ref_chunks,
            "ref_chunks_text": list(extra.get("ref_chunks_text") or []),
            "n_chunks": len(chunk_bodies),
            "context_chars": len(content),
            "approx_context_tokens": int(len(content) / CHARS_PER_TOKEN),
            "subset": subset,
            "split": split,
        }
        return {
            "example_id": 0,
            "prompt": [{"role": "user", "content": question}],
            "answer": json.dumps(
                {"answer": str(gold), "useful_chunks": ref_chunks},
                ensure_ascii=False,
            ),
            "info": json.dumps(info, ensure_ascii=False),
        }

    streams: list[tuple[str, Iterator[dict]]] = []
    for subset in subsets:
        key = (subset, split)
        if key not in _FILES:
            raise ValueError(f"no LongRLVR file for subset={subset!r} split={split!r}")
        path = hf_hub_download(_REPO, _FILES[key], repo_type="dataset")
        streams.append((subset, _iter_records(path, seed)))

    limit = num_examples if (num_examples and num_examples > 0) else None
    while streams and (limit is None or len(rows) < limit):
        still_live: list[tuple[str, Iterator[dict]]] = []
        for subset, stream in streams:
            if limit is not None and len(rows) >= limit:
                still_live.append((subset, stream))
                continue
            try:
                rec = next(stream)
            except StopIteration:
                continue
            scanned += 1
            row = _process(rec, subset)
            if row is not None:
                rows.append(row)
            still_live.append((subset, stream))
        streams = still_live
    for i, row in enumerate(rows):
        row["example_id"] = i

    if stats is not None:
        stats.update({"scanned": scanned, "kept": len(rows), "drops": drops})
    if not rows:
        raise ValueError(
            f"No LongRLVR examples for subsets={list(subsets)} split={split!r} "
            f"length={length!r}. Drops: {drops}"
        )
    return Dataset.from_list(rows)


def load_environment(
    *,
    subset: str = "llama",
    split: str = "train",
    length: str | Sequence[str] | None = None,
    num_examples: int = 2000,
    seed: int = 42,
    answer_metric: str = "judge",
    judge_model: str = "openai/gpt-4.1-mini",
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_subcall: int = 0,
    **kwargs: Any,
) -> vf.Environment:
    """Load LongRLVR wired through ``RLMTrainEnv``."""
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
    subsets = [subset]

    correctness, metrics = make_reward(answer_metric=answer_metric, judge_model=judge_model)
    stats: dict[str, Any] = {}
    dataset = _build_dataset(
        subsets=subsets,
        split=split,
        length=length,
        num_examples=num_examples,
        seed=seed,
        stats=stats,
    )
    rubric = rlm_train.RLMTrainRubric(
        correctness=correctness, weight=1.0, min_subcall=min_subcall,
    )
    for m in metrics:
        rubric.add_metric(m)
    env = rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )
    env.longrlvr_build_stats = stats  # type: ignore[attr-defined]
    return env


__all__ = [
    "load_environment",
    "make_reward",
    "parse_upstream_prompt",
    "parse_final_answer",
    "context_scores",
    "combine",
    "parse_judgment",
    "ParseError",
    "SUBSETS",
    "SPLITS",
    "LENGTH_BUCKETS",
    "ANSWER_METRICS",
    "MAX_ROOT_PROMPT_CHARS",
    "CHARS_PER_TOKEN",
]
