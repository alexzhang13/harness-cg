"""Real CTC-Bench split_v1 rows (cached per process) and the sidecar facts the oracles need.

Sidecars: ``outlier_articles.json.gz`` (source article of each outlier document) and ``reorder_seams.json`` (seam
positions in reorder rows). A missing sidecar is an error.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from functools import cache
from pathlib import Path
from typing import Any

from ctcbench_real import env as R

FAMILIES = ("textgroups", "hotpotqa", "nq", "outlier_fixedM", "outlier", "grouping", "reorder", "qdmatch_hpqa",
            "contradiction_iid")
ALIASES = {"contradiction": "contradiction_iid", "hpqa": "hotpotqa", "outlier_fixedm": "outlier_fixedM",
           "qdmatch": "qdmatch_hpqa"}
SIDECARS = Path(__file__).with_name("sidecars")
TRAIN_RUNGS = ("r8k", "r16k", "r32k")


def resolve(family: str) -> str:
    f = ALIASES.get(family, family)
    if f not in FAMILIES:
        raise ValueError(f"unknown CTC family {family!r}; expected one of {FAMILIES}")
    return f


def spec_of(family: str):
    if family == "xabsence_paraphrase":
        from ctcbench_real.core import spec_of as core_spec
        return core_spec(family)
    return R._registry.get(R.ROSTER[resolve(family)].spec)


def row_key(ex: dict) -> str:
    """Content key of a row (stable across processes): sha1 over its document texts."""
    h = hashlib.sha1()
    for d in ex.get("documents") or []:
        h.update((d.get("text") or "").encode("utf-8", "ignore"))
        h.update(b"\x1e")
    return h.hexdigest()[:20]


def text_key(text: str) -> str:
    return hashlib.sha1(" ".join((text or "").split()).encode("utf-8", "ignore")).hexdigest()[:20]


@cache
def examples(family: str, split: str, decontaminate: bool = True) -> tuple[dict, ...]:
    """split_v1 rows of ``family`` / ``split``, minus any row overlapping the eval set at any rung."""
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    subset = R.resolve_task(resolve(family))
    spec = R._registry.get(R.ROSTER[subset].spec)
    train_dir = os.environ.get(R.TRAIN_DIR_ENV) or R.DEFAULT_TRAIN_DIR
    exs = R._train_rows(subset, train_dir, TRAIN_RUNGS)
    exs = R._versioned(exs, subset, spec, split, "v1")
    if spec.primary_metric == "pairwise_f1":
        exs = [ex for ex in exs if any(len(g) > 1 for g in (R._gold_of(ex, spec) or []))]
    if decontaminate:
        from . import decontam
        if decontam.enabled():
            exs, _counts = decontam.filter_rows(resolve(family), exs)
    return tuple(sorted(exs, key=row_key))


def render(ex: dict, family: str) -> tuple[str, str]:
    """(context, root_prompt) exactly as the real env renders the row."""
    context, task_text, _q = R.render(ex, spec_of(family))
    return context, f"{task_text}\n\n{R.submit_hint(task_text)}"


def score(answer: Any, ex: dict, family: str) -> dict[str, float]:
    """The benchmark's scorer: all metrics plus ``primary`` and ``parse_ok``."""
    slim = {k: v for k, v in ex.items() if k != "documents"}
    slim["n_docs"] = len(ex["documents"]) if ex.get("documents") else int(ex.get("n_docs") or 0)
    return R.score_answer(answer, slim, spec_of(family))


def gold_target(ex: dict, family: str) -> str:
    return R.gold_target(ex, spec_of(family))


@cache
def sidecar(name: str) -> dict:
    path = SIDECARS / name
    if not path.exists():
        if os.environ.get("CTCBENCH_SIDECAR_BUILD"):
            return {}  # build_sidecars.py reads rows before the sidecar exists
        raise FileNotFoundError(f"{path} missing: build it with environments/synthetic/ctcbench/tools/build_sidecars.py")
    if name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text())


def meta(ex: dict) -> dict:
    m = ex.get("_meta") or ex.get("meta") or {}
    if isinstance(m, str):
        try:
            m = json.loads(m)
        except json.JSONDecodeError:
            m = {}
    return m


__all__ = ["ALIASES", "FAMILIES", "examples", "gold_target", "meta", "render", "resolve", "row_key", "score",
           "sidecar", "spec_of", "text_key"]
