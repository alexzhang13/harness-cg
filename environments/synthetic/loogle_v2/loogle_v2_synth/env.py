"""Procedural LooGLE-v2-style documents (finance 10-Ks, code repos, game logs) with the real eval's answer formats
and scoring."""

from __future__ import annotations

from typing import Any

from synth_common.envkit import Generated

from .finance import FinParams, generate_finance
from .scoring import score_answer


def make_generator(family: str, fin: FinParams | None = None, code_kw: dict | None = None,
                   game_kw: dict | None = None):
    def generate(info: dict) -> Generated:
        if family == "finance":
            return generate_finance(info, fin)
        if family == "code":
            from .code import generate_code
            return generate_code(info, **(code_kw or {}))
        if family == "game":
            from .game import generate_game
            return generate_game(info, **(game_kw or {}))
        raise ValueError(f"unknown family {family!r}")

    return generate


def score(final: str, gold: Any, hidden: dict, state: Any) -> float:
    return score_answer(final, str(gold), hidden.get("answer_type", "QA"), hidden.get("options") or [])


__all__ = ["make_generator", "score"]
