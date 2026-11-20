"""OOLONG-real answer parsing and scoring."""

from __future__ import annotations

import re

_BOX_TEXT = re.compile(r"\\boxed\{\\text\{([^}]*)\}\}")
_BOX = re.compile(r"\\boxed[\{]+([^}]*)[\}]+")


def extract_answer(text: str) -> str:
    text = (text or "").strip()
    m = _BOX_TEXT.search(text) or _BOX.search(text)
    if m:
        return m.group(1)
    return text


def parse_answer(answer: str) -> int | str | list[str]:
    """Verbatim ``dnd_parse_answer``."""
    try:
        return int(answer)
    except ValueError:
        pass
    if "," in answer:
        return [item.strip() for item in answer.split(",") if item.strip()]
    return answer


def _norm_pred(final: str) -> int | str | list[str]:
    s = extract_answer(final).strip()
    s = s.strip("`\"' ").rstrip(".")
    return parse_answer(s)


def score_paper(final: str, gold: str) -> float:
    g = parse_answer(str(gold))
    p = _norm_pred(final)
    if isinstance(g, int) and isinstance(p, int):
        return 0.75 ** abs(g - p)
    if isinstance(g, str) and isinstance(p, str):
        return float(g.strip().lower() == p.strip().lower())
    if isinstance(g, list) and isinstance(p, list):
        return len(set(g) & set(p)) / len(set(g)) if g else 0.0
    return 0.0


def score_synth(final: str, gold: str) -> float:
    g = parse_answer(str(gold))
    p = _norm_pred(final)
    if isinstance(g, list) and isinstance(p, list):
        gs = {x.strip().lower() for x in g}
        ps = {x.strip().lower() for x in p}
        return len(gs & ps) / max(len(gs), len(ps)) if gs else 0.0
    return score_paper(final, gold)


def abs_err(final: str, gold: str) -> float:
    g = parse_answer(str(gold))
    p = _norm_pred(final)
    if isinstance(g, int) and isinstance(p, int):
        return float(abs(g - p))
    return -1.0


__all__ = ["abs_err", "extract_answer", "parse_answer", "score_paper", "score_synth"]
