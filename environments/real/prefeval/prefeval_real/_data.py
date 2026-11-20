"""PrefEval raw-data acquisition + cache."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from functools import cache
from pathlib import Path

from ._upstream import ALL_TOPICS

UPSTREAM_SHA = "50795054b5ff5f418d2b768a331d71e480f93331"
_RAW = f"https://raw.githubusercontent.com/amazon-science/PrefEval/{UPSTREAM_SHA}"

PREF_FORM_DIR: dict[str, str] = {
    "explicit": "benchmark_dataset/explicit_preference",
    "implicit_persona": "benchmark_dataset/implicit_preference/persona-driven",
}
PREF_FORMS: tuple[str, ...] = tuple(PREF_FORM_DIR)

MCQ_DIR = "benchmark_dataset/mcq_options"
INTER_TURNS_PATH = "benchmark_dataset/filtered_inter_turns.json"


def cache_root() -> Path:
    """Where fetched files land (or where a local clone lives)."""
    local = os.environ.get("PREFEVAL_REPO_DIR")
    if local:
        return Path(local)
    base = os.environ.get("PREFEVAL_DATA_DIR")
    if base:
        return Path(base)
    return Path.home() / ".cache" / "prefeval" / UPSTREAM_SHA[:7]


def _fetch(rel_path: str, *, timeout: float = 60.0) -> Path:
    """Return a local path for ``rel_path``, downloading it if absent."""
    dest = cache_root() / rel_path
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if os.environ.get("PREFEVAL_REPO_DIR"):
        raise FileNotFoundError(
            f"PREFEVAL_REPO_DIR={os.environ['PREFEVAL_REPO_DIR']!r} is set but "
            f"{dest} is missing. Either unset it (to download) or clone "
            "https://github.com/amazon-science/PrefEval there."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_RAW}/{rel_path}"
    tmp = dest.with_suffix(dest.suffix + f".part{os.getpid()}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            tmp.write_bytes(resp.read())
    except urllib.error.HTTPError as e:  # pragma: no cover
        raise RuntimeError(
            f"PrefEval download failed for {url} (HTTP {e.code}). The upstream "
            f"layout may have moved off commit {UPSTREAM_SHA[:7]}."
        ) from e
    except OSError as e:  # pragma: no cover
        raise RuntimeError(
            f"PrefEval download failed for {url}: {e}. This env is setup tier C "
            "(plain HTTPS); on an air-gapped node clone the repo and set "
            "PREFEVAL_REPO_DIR."
        ) from e
    os.replace(tmp, dest)
    return dest


def _load_json(rel_path: str):
    with open(_fetch(rel_path)) as fh:
        return json.load(fh)


@cache
def load_pref_topic(pref_form: str, topic: str) -> tuple[dict, ...]:
    """The per-topic record list for one preference form."""
    if pref_form not in PREF_FORM_DIR:
        raise ValueError(f"pref_form must be one of {PREF_FORMS}, got {pref_form!r}")
    if topic not in ALL_TOPICS:
        raise ValueError(f"unknown topic {topic!r}; choose from {ALL_TOPICS}")
    return tuple(_load_json(f"{PREF_FORM_DIR[pref_form]}/{topic}.json"))


@cache
def load_mcq_topic(topic: str) -> tuple[dict, ...]:
    """The per-topic classification-task options."""
    if topic not in ALL_TOPICS:
        raise ValueError(f"unknown topic {topic!r}; choose from {ALL_TOPICS}")
    return tuple(_load_json(f"{MCQ_DIR}/{topic}.json"))


@cache
def load_distractor_messages() -> tuple[dict, ...]:
    """The flat, ordered filler-message stream used for turn distance."""
    raw = _load_json(INTER_TURNS_PATH)
    msgs: list[dict] = []
    for conv in raw:
        for m in conv["conversation"]:
            msgs.append({"role": m["role"], "content": m["content"]})
    for i, m in enumerate(msgs):
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected:
            raise RuntimeError(
                f"filtered_inter_turns.json message {i} has role {m['role']!r}, "
                f"expected {expected!r} — upstream data layout changed"
            )
    return tuple(msgs)


MAX_INTER_TURNS = 316


__all__ = [
    "UPSTREAM_SHA",
    "PREF_FORM_DIR",
    "PREF_FORMS",
    "MAX_INTER_TURNS",
    "cache_root",
    "load_pref_topic",
    "load_mcq_topic",
    "load_distractor_messages",
]
