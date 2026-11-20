"""Verbatim upstream PrefEval artefacts (topic list, MCQ question format, option shuffle, choice extractor).

Copied from ``amazon-science/PrefEval`` @ commit ``50795054b5ff5f418d2b768a331d71e480f93331``
(``utils/common_utils.py`` and ``utils/utils_mcq.py``).
"""

from __future__ import annotations

import re
from typing import Any


ALL_TOPICS: tuple[str, ...] = (
    "travel_transportation",
    "shop_motors",
    "lifestyle_beauty",
    "travel_restaurant",
    "shop_fashion",
    "entertain_shows",
    "pet_ownership",
    "lifestyle_fit",
    "entertain_games",
    "shop_home",
    "lifestyle_health",
    "travel_activities",
    "education_learning_styles",
    "entertain_music_book",
    "professional_work_location_style",
    "education_resources",
    "lifestyle_dietary",
    "shop_technology",
    "travel_hotel",
    "entertain_sports",
)

TOPIC_DOMAIN: dict[str, str] = {t: t.split("_", 1)[0] for t in ALL_TOPICS}


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def format_options(options: list[str]) -> str:
    """utils/utils_mcq.py::format_options"""
    return "\n".join(f"{_LETTERS[i]}. {option}" for i, option in enumerate(options))


def get_mcq_question_format(options: list[str]) -> str:
    """utils/utils_mcq.py::get_mcq_question_format (byte-for-byte)."""
    formatted_options = format_options(options)
    mcq_question_format = f"""
    I'm trying to decide on this and here are 4 options for my query: \n{formatted_options}\nNow, I'd like you to pick one of them as your top recommendation for me.
    Important instructions for your response:
    1. Choose only one option (A, B, C, or D) that best matches my preferences.
    2. Your answer must be one of these options.
    3. Don't say things like "I can't choose" or suggest alternatives not listed.
    4. Answer example: <choice>B</choice>. Give me your answer in this exact format, without any additional explanation:
       <choice>[A/B/C/D]</choice>
    """
    return mcq_question_format


def shuffle_options(options: list[str], rng: Any) -> tuple[list[str], int]:
    """utils/utils_mcq.py::shuffle_options, with an injected ``rng`` so each row is reproducible."""
    correct_answer = options[0]
    shuffled_options = rng.sample(options, len(options))
    correct_index = shuffled_options.index(correct_answer)
    return shuffled_options, correct_index


_CHOICE_TAG_RE = re.compile(r"<\s*choice\s*>(.*?)(?:<\s*/\s*choice\s*>|$)", re.IGNORECASE | re.DOTALL)


def extract_choice(response: str | None) -> str | None:
    """utils/utils_mcq.py::extract_choice (bs4-free port)."""
    if not response:
        return None
    m = _CHOICE_TAG_RE.search(response)
    if not m:
        return None
    inner = m.group(1)
    letter = re.search(r"[ABCD]", inner)
    return letter.group(0) if letter else None


__all__ = [
    "ALL_TOPICS",
    "TOPIC_DOMAIN",
    "format_options",
    "get_mcq_question_format",
    "shuffle_options",
    "extract_choice",
]
