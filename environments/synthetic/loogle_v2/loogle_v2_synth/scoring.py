"""Answer scoring copied from ``loogle_v2_real`` so the synthetic reward matches the eval without loading HF data."""

from __future__ import annotations

import ast
import json
import re
import string
from typing import Any

_PUNCT = str.maketrans("", "", string.punctuation.replace("_", ""))


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.translate(_PUNCT)
    return re.sub(r"\s+", " ", s).strip()


_SINGLE_LETTER_RE = re.compile(r"^[A-Ha-h]$")


def match_single_letter(pred: str, gold: str) -> float:
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


def exact_or_token_match(pred: str, gold: str) -> float:
    if _SINGLE_LETTER_RE.match((gold or "").strip()):
        return match_single_letter(pred, gold)
    np_, ng = _norm(pred), _norm(gold)
    if not ng:
        return 0.0
    if np_ == ng:
        return 1.0
    if re.search(rf"(?<![\w]){re.escape(ng)}(?![\w])", np_):
        return 1.0
    return 0.0


_OPTION_PREFIX = re.compile(r"^\s*\(?([A-Ha-h])\)?[.):]\s*")


def option_texts(options: list[str]) -> dict[str, str]:
    out = {}
    for o in options:
        m = _OPTION_PREFIX.match(o)
        if m:
            out[m.group(1).upper()] = o[m.end():].strip()
    return out


def match_option_text(pred: str, gold_letter: str, options: dict[str, str]) -> float:
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


def as_list(s: Any) -> list[str] | None:
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


_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9, "bn": 1e9, "t": 1e12}
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_MAX_PRED_NUMBERS = 8


def _numbers_with_scale(s: Any, limit: int = 1) -> list[tuple[float, float]]:
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
        if not tail.startswith("%"):
            for suf, mult in sorted(_SUFFIX.items(), key=lambda kv: -len(kv[0])):
                if tail.startswith(suf):
                    scale = mult
                    break
        out.append((mantissa, mantissa * scale))
    return out


_YEAR_PAIR_RE = re.compile(r"(?<!\w)(?:FY\s*)?(\d{4})\s*(?:[-–—→]|to)\s*(?:FY\s*)?(\d{4})(?!\w)", re.I)
_ANSWER_PREFIX_RE = re.compile(r"\b(?:the\s+)?(?:(?:correct|final)\s+)?answer\s*(?:is\b\s*[:：]?|[:：])\s*", re.I)


def numeric_match(pred: Any, gold: Any, tol: float) -> float | None:
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

    return 1.0 if any(close(a, b) for pair in p for a in pair for b in g[0]) else 0.0


def score_answer(pred: str, gold: str, answer_type: str, options: list[str] | None = None, tol: float = 1e-3) -> float:
    """Same dispatch as the real env's ``_score`` (minus the judge fallback, which no synthetic row needs)."""
    pred = pred or ""
    gold = str(gold or "")
    if not pred.strip() or not gold.strip():
        return 0.0
    gl = as_list(gold)
    if gl is not None:
        from_pred = _parse_pred_list(pred)
        return _set_f1(from_pred, gl)
    if answer_type == "MC":
        hit = exact_or_token_match(pred, gold)
        if hit == 0.0 and _SINGLE_LETTER_RE.match(gold.strip()):
            hit = match_option_text(pred, gold, option_texts(options or []))
        return hit
    tolerance = 0.0 if re.fullmatch(r"[+-]?\d+", gold.strip()) else tol
    num = numeric_match(pred, gold, tolerance)
    if num is not None:
        return num
    return exact_or_token_match(pred, gold)


_ITEM_SEP_RE = re.compile(r"[,，;；]")
_FILE_LIST_PREFIX_RE = re.compile(r"\A\s*(?:the\s+)?changed\s+files\s+are\s*[:：]?\s*", re.I)


def _clean_item(s: str) -> str:
    return (s or "").strip().strip(" `'\"[]")


def _parse_pred_list(pred: str) -> list[str]:
    pred = _FILE_LIST_PREFIX_RE.sub("", pred or "", count=1)
    got = as_list(pred)
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
