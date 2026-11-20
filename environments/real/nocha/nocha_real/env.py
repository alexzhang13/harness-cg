"""NoCha (`nocha-real`): TRUE / FALSE claim verification over a whole book (public classics sample)."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import verifiers as vf
from datasets import Dataset

import rlm_train


_NOCHA_REPO_RAW = "https://raw.githubusercontent.com/marzenakrp/nocha/main"
_DATA_SAMPLE_URL = f"{_NOCHA_REPO_RAW}/data_sample/data_sample.json"
_CLASSIC_BOOKS_URL = f"{_NOCHA_REPO_RAW}/data_sample/classic_books.pkl"
_CACHE_DIR = Path.home() / ".cache" / "rlm_train_real" / "nocha"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "rlm-train-real"})
    with urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def _load_samples() -> list[dict]:
    data_path = _CACHE_DIR / "data_sample.json"
    books_path = _CACHE_DIR / "classic_books.pkl"
    if not data_path.exists():
        _download(_DATA_SAMPLE_URL, data_path)
    if not books_path.exists():
        _download(_CLASSIC_BOOKS_URL, books_path)
    with open(data_path) as f:
        raw = json.load(f)
    with open(books_path, "rb") as f:
        books = pickle.load(f)
    out: list[dict] = []
    for i, row in enumerate(raw):
        title = row.get("book_title", "")
        out.append({
            "id": i,
            "claim": row.get("claim", ""),
            "type": row.get("type"),
            "context": books.get(title, ""),
            "book_title": title,
        })
    return out


def _parse_true_false(s: str) -> bool | None:
    if not s:
        return None
    m = re.search(r"<answer>\s*(TRUE|FALSE)\s*</answer>", s, re.IGNORECASE)
    if m:
        return m.group(1).upper() == "TRUE"
    for part in reversed(s.split()):
        t = re.sub(r"[^\w]", "", part).upper()
        if t == "TRUE":
            return True
        if t == "FALSE":
            return False
    return None


async def _score(answer: str, state: vf.State, **_kw: Any) -> float:
    final = state.get("rlm_final_answer") or state.get("final_answer") or ""
    gold = (answer or "").strip().upper() == "TRUE"
    pred = _parse_true_false(final)
    if pred is None:
        return 0.0
    return 1.0 if pred == gold else 0.0


def _build_dataset(num_examples: int, seed: int) -> Dataset:
    samples = _load_samples()
    rows: list[dict] = []
    for s in samples:
        gold_type = s["type"]
        if isinstance(gold_type, bool):
            answer = "TRUE" if gold_type else "FALSE"
        else:
            answer = str(gold_type).strip().upper()
        title = s.get("book_title", "")
        claim = s["claim"]
        info = {
            "context": s["context"],
            "root_prompt": (
                f"Claim: {claim}\n"
                f"(The book {title!r} is in `context`. Answer TRUE or FALSE.)"
            ),
            "book_title": title,
            "claim": claim,
        }
        rows.append({
            "example_id": s["id"],
            "prompt": [{"role": "user", "content": claim}],
            "answer": answer,
            "info": json.dumps(info),
        })
    out = Dataset.from_list(rows)
    if seed is not None:
        out = out.shuffle(seed=seed)
    if num_examples and num_examples > 0:
        out = out.select(range(min(num_examples, len(out))))
    return out


def load_environment(
    *,
    num_examples: int = -1,
    seed: int = 42,
    max_iterations: int = 12,
    sub_max_tokens: int = 4096,
    min_subcall: int = 1,
    **kwargs: Any,
) -> vf.Environment:
    dataset = _build_dataset(num_examples, seed)
    rubric = rlm_train.RLMTrainRubric(correctness=_score, weight=1.0, min_subcall=min_subcall)
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = ["load_environment"]
