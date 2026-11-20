"""LooGLE-v2 (`loogle-v2-real`): questions over long real documents (court opinions, 10-Ks, repos, game logs).

By default serves the frozen 206-row eval cohort (`eval_indices.json`, documents >= 2M characters). Scoring depends
on the gold: set-F1 for lists, letter / text match for MC, exact or tolerance-based numeric match for QA.
"""

from __future__ import annotations

import ast
import json
import re
import string
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

import rlm_train

HF_DATASET = "MuLabPKU/LooGLE-v2"
# Prompt / scoring version; load_environment checks it.
EVALUATION_VERSION = "2026-10-11"

SOURCES = ("Law", "Finance", "Code", "Game")
TYPES = ("MC", "QA", "SA")

_CONTEXT_POINTER = (
    "[The full document is NOT reproduced here. It is available in the REPL as "
    "the `context` string variable; query it with `llm_query` over slices.]"
)

_PUNCT = str.maketrans("", "", string.punctuation.replace("_", ""))


def _norm(s: str) -> str:
    """Lowercase, strip punctuation (keeping ``_``), collapse whitespace."""
    s = (s or "").strip().lower()
    s = s.translate(_PUNCT)
    return re.sub(r"\s+", " ", s).strip()


_SINGLE_LETTER_RE = re.compile(r"^[A-Ha-h]$")


def _match_single_letter(pred: str, gold: str) -> float:
    """Match a bare option letter only in an explicit answer position ("D", "answer: D", "(D)"), not the word "a"."""
    want = gold.strip().upper()
    s = (pred or "").strip().strip("`'\" .")
    if s.upper() == want:
        return 1.0
    for pat in (
        r"(?:answer|letter|choice|option|final)\b"
        r"(?:\s+(?:is|was|are|would\s+be|should\s+be))?"
        r"\s*[:\-]?\s*[(\[]?([A-H])\b",
        r"\(([A-H])\)",
        r"\[([A-H])\]",
        r"^\s*([A-H])[.):]",
    ):
        m = re.search(pat, s, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return 1.0 if m.group(1).upper() == want else 0.0
    return 0.0


def _exact_or_token_match(pred: str, gold: str) -> float:
    """1.0 if `pred` equals `gold`, or contains it as a whole token."""
    if _SINGLE_LETTER_RE.match((gold or "").strip()):
        return _match_single_letter(pred, gold)
    np_, ng = _norm(pred), _norm(gold)
    if not ng:
        return 0.0
    if np_ == ng:
        return 1.0
    if re.search(rf"(?<![\w]){re.escape(ng)}(?![\w])", np_):
        return 1.0
    return 0.0


def _as_list(s: Any) -> list[str] | None:
    """Parse a Python/JSON list literal, or None if `s` isn't one."""
    if isinstance(s, list):
        return [str(x) for x in s]
    txt = str(s or "").strip()
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(m.group(0))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, list):
            return [str(x) for x in v]
    return None


_ITEM_SEP_RE = re.compile(r"[,，;；]")
_FILE_LIST_PREFIX_RE = re.compile(r"\A\s*(?:the\s+)?changed\s+files\s+are\s*[:：]?\s*", re.I)


def _parse_pred_list(pred: str) -> list[str]:
    """Items from a prediction: list literal, bracket contents, lines, or separator-split."""
    pred = _FILE_LIST_PREFIX_RE.sub("", pred or "", count=1)
    got = _as_list(pred)
    if got is not None:
        return got

    m = re.search(r"\[(.*)\]", pred or "", re.S)
    if m and m.group(1).strip():
        items = [_clean_item(p) for p in _ITEM_SEP_RE.split(m.group(1))]
        items = [i for i in items if i]
        if items:
            return items

    lines = [ln.strip() for ln in (pred or "").splitlines() if ln.strip()]
    if len(lines) > 1:
        out = [_clean_item(re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln)) for ln in lines]
        return [o for o in out if o]

    if _ITEM_SEP_RE.search(pred or ""):
        out = [_clean_item(p) for p in _ITEM_SEP_RE.split(pred)]
        return [o for o in out if o]
    return [pred.strip()] if (pred or "").strip() else []


def _clean_item(s: str) -> str:
    return (s or "").strip().strip(" `'\"[]")


def _set_f1(pred_items: list[str], gold_items: list[str]) -> float:
    gold = {_norm(g) for g in gold_items if _norm(g)}
    pred = {_norm(p) for p in pred_items if _norm(p)}
    if not gold:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(gold & pred)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(pred), tp / len(gold)
    return 2 * precision * recall / (precision + recall)


_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9, "bn": 1e9, "t": 1e12}
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


_MAX_PRED_NUMBERS = 8


def _numbers_with_scale(s: Any, limit: int = 1) -> list[tuple[float, float]]:
    """``(mantissa, scaled)`` for up to `limit` numbers in `s`."""
    txt = str(s if s is not None else "").strip().lower().replace(" ", "")
    if not txt:
        return []
    out: list[tuple[float, float]] = []
    for m in _NUM_RE.finditer(txt):
        if len(out) >= limit:
            break
        try:
            mantissa = float(m.group(0).replace(",", ""))
        except ValueError:
            continue
        tail = txt[m.end():]
        scale = 1.0
        if tail.startswith("%"):
            scale = 1.0
        else:
            for suf, mult in sorted(_SUFFIX.items(), key=lambda kv: -len(kv[0])):
                if tail.startswith(suf):
                    scale = mult
                    break
        out.append((mantissa, mantissa * scale))
    return out


_YEAR_PAIR_RE = re.compile(
    r"(?<!\w)(?:FY\s*)?(\d{4})\s*(?:[-–—→]|to)\s*(?:FY\s*)?(\d{4})(?!\w)", re.I,
)
_ANSWER_PREFIX_RE = re.compile(r"\b(?:the\s+)?(?:(?:correct|final)\s+)?answer\s*(?:is\b\s*[:：]?|[:：])\s*", re.I)


def _numeric_match(pred: Any, gold: Any, tol: float) -> float | None:
    """1.0/0.0 on a numeric comparison, or None if either side has no number."""
    gold_pair = _YEAR_PAIR_RE.fullmatch(str(gold).strip())
    if gold_pair:
        answer_text = _ANSWER_PREFIX_RE.split(str(pred))[-1]
        pairs = {tuple(map(int, m.groups())) for m in _YEAR_PAIR_RE.finditer(answer_text)}
        return float(pairs == {tuple(map(int, gold_pair.groups()))})

    g = _numbers_with_scale(gold, limit=1)
    p = _numbers_with_scale(pred, limit=_MAX_PRED_NUMBERS)
    if not g or not p:
        return None

    def close(a: float, b: float) -> bool:
        if b == 0:
            return abs(a) <= tol
        return abs(a - b) / abs(b) <= tol

    return 1.0 if any(close(a, b) for pair in p for a in pair
                      for b in g[0]) else 0.0


def _make_score(num_tolerance: float):
    async def _score(info, state: vf.State, **_kw: Any) -> float:
        meta = json.loads(info) if isinstance(info, str) else info
        gold = str(meta.get("gold_answer") or "")
        qtype = (meta.get("answer_type") or "").upper()
        final = state.get("rlm_final_answer") or state.get("final_answer") or ""
        pred = final if isinstance(final, str) else str(final)
        if not pred.strip() or not gold.strip():
            return 0.0

        # list-valued golds (Version Control rows) use set-F1
        gold_list = _as_list(gold)
        if gold_list is not None:
            return _set_f1(_parse_pred_list(pred), gold_list)

        if qtype == "MC":
            hit = _exact_or_token_match(pred, gold)
            if hit == 0.0 and _SINGLE_LETTER_RE.match(gold.strip()):
                hit = _match_option_text(pred, gold, _option_texts(meta.get("options") or []))
            return hit

        if qtype == "SA":
            return _exact_or_token_match(pred, gold)

        # QA: numeric match first (integers exact), then string match
        tolerance = 0.0 if re.fullmatch(r"[+-]?\d+", gold.strip()) else num_tolerance
        num = _numeric_match(pred, gold, tolerance)
        if num is not None:
            return num
        if _exact_or_token_match(pred, gold) == 1.0:
            return 1.0
        state["loogle_qa_unparsed"] = True
        return 0.0

    return _score


def _parse_options(options_raw: Any) -> list[str]:
    """The ``options`` column as a list (stored as JSON or as a Python repr upstream)."""
    if isinstance(options_raw, str):
        s = options_raw.strip()
        if not s:
            return []
        try:
            options = json.loads(s)
        except json.JSONDecodeError:
            try:
                options = ast.literal_eval(s)
            except (ValueError, SyntaxError) as exc:
                raise ValueError("LooGLE options must be a JSON or Python list of strings") from exc
    else:
        options = options_raw or []
    if not isinstance(options, (list, tuple)) or any(not isinstance(o, str) or not o.strip() for o in options):
        raise ValueError("LooGLE options must be a list of nonempty strings")
    return list(options)


def _format_options(options_raw: Any) -> str:
    """Options appended after the question (used when the instruction has no ``{options}`` slot)."""
    options = _parse_options(options_raw)
    if not options:
        return ""
    return "\n\nOptions:\n" + "\n".join(options)


_OPTION_PREFIX = re.compile(r"^\s*\(?([A-Ha-h])\)?[.):]\s*")


def _option_texts(options: list[str]) -> dict[str, str]:
    """{letter: option text without its "A. " prefix} for options that carry a letter prefix."""
    out = {}
    for o in options:
        m = _OPTION_PREFIX.match(o)
        if m:
            out[m.group(1).upper()] = o[m.end():].strip()
    return out


def _match_option_text(pred: str, gold_letter: str, options: dict[str, str]) -> float:
    """Credit an answer that gives the gold option's text instead of its letter, unless it also names another option."""
    want = options.get(gold_letter.strip().upper())
    if not want:
        return 0.0
    np_, nw = _norm(pred), _norm(want)
    if not nw:
        return 0.0
    if np_ == nw:
        return 1.0
    def has(t: str) -> bool:
        return bool(t) and re.search(rf"(?<![\w]){re.escape(t)}(?![\w])", np_) is not None
    if has(nw) and not any(has(_norm(t)) for k, t in options.items() if k != gold_letter.strip().upper()):
        return 1.0
    return 0.0


# rough chars-per-token for the length filters
_CHARS_PER_TOKEN = 4

# Frozen eval cohort: test-split positions of the 206 rows with >= 2M-character contexts.
EVAL_INDICES_FILE = Path(__file__).with_name("eval_indices.json")


def eval_indices() -> list[int]:
    return list(json.loads(EVAL_INDICES_FILE.read_text())["indices"])


def _build_dataset(
    *, indices: Sequence[int] | None, source: str | None, task: str | None, answer_type: str | None,
    num_examples: int, seed: int | None, max_context_chars: int | None,
    min_context_tokens: int | None, max_context_tokens: int | None,
) -> Dataset:
    ds = load_dataset(HF_DATASET, split="test")
    if indices is not None:
        ds = ds.select(list(indices))
    if source:
        ds = ds.filter(lambda r: (r.get("source") or "") == source)
    if task:
        ds = ds.filter(lambda r: (r.get("task") or "") == task)
    if answer_type:
        ds = ds.filter(lambda r: (r.get("type") or "").upper() == answer_type.upper())
    # length bands use the untruncated context
    if min_context_tokens is not None:
        lo = min_context_tokens * _CHARS_PER_TOKEN
        ds = ds.filter(lambda r: len(r.get("context") or "") >= lo)
    if max_context_tokens is not None:
        hi = max_context_tokens * _CHARS_PER_TOKEN
        ds = ds.filter(lambda r: len(r.get("context") or "") <= hi)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if num_examples and num_examples > 0:
        ds = ds.select(range(min(num_examples, len(ds))))

    rows: list[dict] = []
    for i, r in enumerate(ds):
        ctx = r.get("context") or ""
        question = (r.get("question") or "").strip()
        gold = (r.get("answer") or "").strip()
        if not ctx or not question or not gold:
            continue
        if max_context_chars is not None and len(ctx) > max_context_chars:
            ctx = ctx[:max_context_chars]

        instruction = (r.get("instruction") or "").strip()
        try:
            options = _parse_options(r.get("options"))
            if _SINGLE_LETTER_RE.fullmatch(gold):
                labeled = _option_texts(options)
                if (len(labeled) != len(options) or len(labeled) < 2
                        or not all(labeled.values()) or gold.upper() not in labeled):
                    raise ValueError("letter answer requires distinct labeled choices including the gold label")
        except ValueError as exc:
            raise ValueError(f"Invalid LooGLE options for item {r.get('id')!r}: {exc}") from exc
        # {context} points at the REPL variable; options are appended if the template has no {options} slot
        opt_block = "" if "{options}" in instruction else _format_options(r.get("options"))
        if instruction:
            root_prompt = (
                instruction
                .replace("{context}", _CONTEXT_POINTER)
                .replace("{question}", question)
                .replace("{options}", "\n".join(options))
            )
            if "{question}" not in instruction and question not in root_prompt:
                root_prompt = f"{root_prompt}\n\nQuestion: {question}"
            root_prompt += opt_block
        else:
            root_prompt = f"{question}{opt_block}"
        if "{options}" in root_prompt or any(o not in root_prompt for o in options):
            raise ValueError(f"LooGLE prompt is missing options for item {r.get('id')!r}")

        info = {
            "context": ctx,
            "root_prompt": root_prompt,
            "gold_answer": gold,
            "question": question,
            "answer_type": (r.get("type") or "").upper(),
            "loogle_task": r.get("task") or "",
            "loogle_source": r.get("source") or "",
            "item_id": r.get("id") or "",
            "options": options,
            "evaluation_version": EVALUATION_VERSION,
        }
        rows.append({
            "example_id": i,
            "prompt": [{"role": "user", "content": "Answer the question about the document."}],
            "answer": gold,
            "info": json.dumps(info),
        })
    return Dataset.from_list(rows)


def load_environment(
    *,
    evaluation_version: str = EVALUATION_VERSION,
    indices: Sequence[int] | str | None = "eval",
    source: str | None = None,
    task: str | None = None,
    answer_type: str | None = None,
    num_examples: int = -1,
    seed: int | None = 42,
    max_context_chars: int | None = None,
    min_context_tokens: int | None = None,
    max_context_tokens: int | None = None,
    num_tolerance: float = 1e-3,
    max_iterations: int = 30,
    sub_max_tokens: int = 4096,
    min_subcall: int = 1,
    **kwargs: Any,
) -> vf.Environment:
    """Load LooGLE-v2 as an RLMTrainEnv.

    indices: ``"eval"`` (the frozen cohort), a list of test-split positions, or None for the whole split.
    num_tolerance: relative error allowed on continuous numeric answers.
    """
    if evaluation_version != EVALUATION_VERSION:
        raise ValueError(f"LooGLE evaluation_version={evaluation_version!r} is unsupported; "
                         f"this adapter requires {EVALUATION_VERSION!r}")
    if source and source not in SOURCES:
        raise ValueError(f"source={source!r} unknown. Choose from {SOURCES}.")
    if answer_type and answer_type.upper() not in TYPES:
        raise ValueError(f"answer_type={answer_type!r} unknown. Choose from {TYPES}.")
    if isinstance(indices, str):
        if indices != "eval":
            raise ValueError(f"indices={indices!r} unknown; use 'eval', a list of positions, or None")
        indices = eval_indices()

    dataset = _build_dataset(
        indices=indices,
        source=source, task=task, answer_type=answer_type,
        num_examples=num_examples, seed=seed,
        max_context_chars=max_context_chars,
        min_context_tokens=min_context_tokens,
        max_context_tokens=max_context_tokens,
    )

    rubric = rlm_train.RLMTrainRubric(
        correctness=_make_score(num_tolerance),
        weight=1.0, min_subcall=min_subcall,
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


TASKS = (
    "Call Graph Analysis", "Legal Case Retrieval", "Cross-Company Comparison",
    "Version Control", "Legal Article Extraction", "Environment Understanding",
    "Rule Understanding", "User Behavior Analysis", "Metric Calculation",
    "Trend Analysis",
)

__all__ = ["load_environment", "eval_indices", "SOURCES", "TYPES", "TASKS"]
