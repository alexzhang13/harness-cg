"""FanOutQA scoring: calls the official upstream implementation."""

from __future__ import annotations

import ast
import functools
import importlib.util
import itertools
import json
import re
from typing import Any

METRICS = ("loose",)


@functools.lru_cache(maxsize=1)
def _official() -> tuple[Any, Any, Any]:
    """``(eval.string, eval.utils, fanoutqa.norm)`` from the installed package."""
    import pathlib

    import fanoutqa
    from fanoutqa import norm

    root = pathlib.Path(fanoutqa.__file__).parent

    def _load(name: str, rel: str):
        spec = importlib.util.spec_from_file_location(name, root / rel)
        if spec is None or spec.loader is None:  # pragma: no cover
            raise ImportError(f"cannot load {rel} from {root}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    return _load("_fanoutqa_eval_string", "eval/string.py"), _load(
        "_fanoutqa_eval_utils", "eval/utils.py"
    ), norm


def answer_in_text(reference: Any, candidate: str):
    """Upstream ``fanoutqa.eval.string.answer_in_text``, verbatim."""
    return _official()[0].answer_in_text(reference, candidate)


def str_answer(answer: Any) -> str:
    """Upstream ``fanoutqa.eval.utils.str_answer``, verbatim."""
    return _official()[1].str_answer(answer)


def normalize(text: Any, remove_stopwords: bool = False) -> str:
    """Upstream ``fanoutqa.norm.normalize`` (ftfy + numbers + spacy lemma + punct)."""
    return _official()[2].normalize(text, remove_stopwords=remove_stopwords)


def normalize_no_lemma(text: Any) -> str:
    """``fanoutqa.norm.normalize`` with the ``lemmatize`` step removed."""
    norm = _official()[2]
    out = str(text).lower()
    out = norm.ftfy.fix_text(out)
    out = norm.normalize_numbers(out)
    out = norm.remove_punct(out)
    return norm.normalize_whitespace(out)


def _bounded(needle: str) -> str:
    """``\\b``-style anchors that still work on non-word-character edges."""
    head = r"\b" if (needle[:1].isalnum() or needle[:1] == "_") else r"(?<!\w)"
    tail = r"\b" if (needle[-1:].isalnum() or needle[-1:] == "_") else r"(?!\w)"
    return head + re.escape(needle) + tail


def reference_strings(reference: Any) -> list[str]:
    """The flat list of strings ``answer_in_text`` checks, in its own order."""
    if isinstance(reference, list):
        return list(itertools.chain.from_iterable(reference_strings(a) for a in reference))
    if isinstance(reference, dict):
        return list(
            itertools.chain.from_iterable(
                reference_strings(a) for a in itertools.chain(reference.keys(), reference.values())
            )
        )
    if isinstance(reference, bool):
        return ["yes" if reference else "no"]
    return [reference]


def loose_accuracy(reference: Any, candidate: str) -> float:
    """Proportion of gold reference strings present in ``candidate``."""
    official = answer_in_text(reference, candidate)
    refs = reference_strings(reference)
    if not refs:
        return 0.0
    missing = set(official.missing)
    light_candidate = normalize_no_lemma(candidate)
    found = 0
    for ref in refs:
        if normalize(ref) not in missing:
            found += 1
            continue
        needle = normalize_no_lemma(ref)
        if needle and re.search(_bounded(needle), light_candidate):
            found += 1
    return found / len(refs)


_ANSWER_KEYS = ("answer", "final_answer", "content", "result", "response", "text")


def parse_final_answer(final: Any) -> str:
    """Coerce ``state["rlm_final_answer"]`` into the answer *string* upstream scores."""
    if final is None:
        return ""
    if isinstance(final, dict):
        if len(final) == 1:
            only_key = next(iter(final))
            if isinstance(only_key, str) and only_key.lower() in _ANSWER_KEYS:
                return parse_final_answer(final[only_key])
        return str_answer(final)
    if isinstance(final, (list, tuple)):
        return str_answer(list(final))
    if isinstance(final, bool):
        return str_answer(final)
    if not isinstance(final, str):
        return str(final)

    text = final.strip()
    if text[:1] in "{[":
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(text)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, (dict, list)):
                return parse_final_answer(obj)
    return text


__all__ = [
    "METRICS",
    "answer_in_text",
    "str_answer",
    "normalize",
    "normalize_no_lemma",
    "reference_strings",
    "loose_accuracy",
    "parse_final_answer",
]
