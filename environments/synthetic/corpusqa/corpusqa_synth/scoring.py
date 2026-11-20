"""Answer scoring -- a vendored copy of ``corpusqa_real.env._make_score`` semantics (kept in sync by hand)."""

from __future__ import annotations

import json
import re
import string
from typing import Any

_PUNCT = str.maketrans("", "", string.punctuation)
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_MIN_CONTAINMENT_LEN = 5


def norm_text(s: Any) -> str:
    s = str(s if s is not None else "").strip().lower()
    s = s.translate(_PUNCT)
    return re.sub(r"\s+", " ", s).strip()


def to_number(s: Any) -> float | None:
    if isinstance(s, bool):
        return None
    if isinstance(s, int | float):
        return float(s)
    txt = str(s or "").strip()
    if not txt:
        return None
    m = _NUM_RE.search(txt.replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def parse_list(pred: str) -> list[str]:
    txt = (pred or "").strip()
    m = re.search(r"\[.*\]", txt, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except json.JSONDecodeError:
            pass
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if len(lines) > 1:
        out = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in lines]
        return [o for o in out if o]
    if "," in txt:
        return [p.strip() for p in txt.split(",") if p.strip()]
    return [txt] if txt else []


def _entity_equal(a: str, b: str) -> bool:
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < _MIN_CONTAINMENT_LEN:
        return False
    return short in long


def set_f1(pred_items: list[str], gold_items: list[str]) -> float:
    gold = list(dict.fromkeys(g for g in (norm_text(x) for x in gold_items) if g))
    pred = list(dict.fromkeys(p for p in (norm_text(x) for x in pred_items) if p))
    if not gold:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    unmatched = list(gold)
    tp = 0
    for p in pred:
        for i, g in enumerate(unmatched):
            if _entity_equal(p, g):
                tp += 1
                unmatched.pop(i)
                break
    if tp == 0:
        return 0.0
    precision, recall = tp / len(pred), tp / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_answer(pred: str, gold: Any, kind: str, num_tolerance: float = 1e-4) -> float:
    pred = pred if isinstance(pred, str) else str(pred)
    if not pred.strip():
        return 0.0
    if kind == "list":
        gold_items = gold if isinstance(gold, list) else [gold]
        return set_f1(parse_list(pred), [str(g) for g in gold_items])
    if kind == "number":
        g, p = to_number(gold), to_number(pred)
        if g is None or p is None:
            return 0.0
        if g == 0:
            return 1.0 if abs(p) <= num_tolerance else 0.0
        return 1.0 if abs(p - g) / abs(g) <= num_tolerance else 0.0
    gs, ps = norm_text(gold), norm_text(pred)
    if not gs:
        return 0.0
    if gs == ps:
        return 1.0
    g, p = to_number(gold), to_number(pred)
    if g is not None and p is not None:
        if g == 0:
            return 1.0 if abs(p) <= num_tolerance else 0.0
        return 1.0 if abs(p - g) / abs(g) <= num_tolerance else 0.0
    return 1.0 if re.search(rf"(?<!\w){re.escape(gs)}(?!\w)", ps) else 0.0


OUTPUT_REQUIREMENTS = (
    "Output requirements:\n"
    "1. Your answer can only be one of the following types: a list of strings (including empty list \"[]\"), a single "
    "short string, a number (floating point numbers should retain 2 decimal places, without units), or a percentage "
    "(retain 2 decimal places).\n"
    "2. To output the date, use the format \"YYYY-MM\"\n"
    "3. If multiple files contain a conflicting value for the same item, please use the value from the latest file. \n"
    "4. Please place the final answer at the end, and strictly use the format: The answer is: xxx. \n"
    "Answer format examples:\n"
    "* For a list of strings: The answer is: [\"NETFLIX INC\", \"H&R BLOCK INC\"]\n"
    "* For a single string: The answer is: \"South\"\n"
    "* For a number: The answer is: 1234567.89\n"
    "* For a percentage: The answer is: 12.34%\n"
)

SYSTEM = "You are a professional data analyst, please answer the question accurately."


def root_prompt(question: str, n_docs: int) -> str:
    """The real env's root prompt (corpusqa_real._build_dataset): question, corpus size and the upstream Output
    requirements; the task block is not part of the last document."""
    head = (f"{SYSTEM}\n\nQuestion: {question}\n\n`context` is a list of {n_docs} documents (the source corpus); "
            "the answer requires aggregating across them.")
    return (f"{head}\n\n{OUTPUT_REQUIREMENTS.strip()}\n\nSubmit the final answer through `answer[\"content\"]`, in "
            "exactly the format required above (the \"The answer is:\" prefix is optional).")


def fmt_number(x: float) -> float:
    return round(float(x), 2)


def fmt_percent(x: float) -> str:
    return f"{x:.2f}%"
