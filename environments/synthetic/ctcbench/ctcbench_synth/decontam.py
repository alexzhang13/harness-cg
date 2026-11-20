"""Drop or scrub synthetic rows that overlap any published CTC-Bench eval row (all configs, all rungs).

An item counts as eval if it matches an eval gold item or question exactly, or, for texts of 13+ words, shares a
13-gram (questions) or half its 13-grams (documents). Rows with eval gold items or questions are dropped; rows with
only eval distractors have them removed. ``CTC_SYNTH_DECONTAM=0`` disables the filter.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import os
import re
from functools import cache
from pathlib import Path

import numpy as np

INDEX = Path(__file__).with_name("sidecars") / "eval_index.npz"
WORD = re.compile(r"[a-z0-9]+")
_OUTLIER = ("outlier", "outlier_fixedM")
_SCRUB = ("nq", "hotpotqa", "qdmatch_hpqa", "contradiction_iid")
_ROW_QUESTION = ("nq", "hotpotqa")


def enabled() -> bool:
    return os.environ.get("CTC_SYNTH_DECONTAM", "1") != "0"


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=8).digest(), "little")


def norm(s: str) -> str:
    return " ".join((s or "").split())


def grams(s: str, n: int = 13) -> list[int]:
    w = WORD.findall((s or "").lower())
    return [h64(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)]


@cache
def index() -> dict[str, frozenset[int]]:
    if not INDEX.exists():
        raise FileNotFoundError(f"{INDEX} missing: run environments/synthetic/ctcbench/tools/build_eval_index.py")
    z = np.load(INDEX)
    return {k: frozenset(int(x) for x in z[k]) for k in ("gold", "gold13", "outlier", "rows", "pubmed13")}


def is_eval(text: str, question: bool = False) -> bool:
    """Exact eval gold item / question, or a near-copy of one (see module doc)."""
    ix = index()
    t = norm(text)
    if not t:
        return False
    if h64(t) in ix["gold"]:
        return True
    g = grams(t)
    if not g:
        return False
    hit = sum(x in ix["gold13"] for x in g)
    return hit > 0 if question else hit * 2 >= len(g)


def _gold_positions(ex: dict, family: str) -> list[int]:
    from ctcbench_real import env as R

    from .data import spec_of
    spec = spec_of(family)
    n = len(ex.get("documents") or [])
    if spec.name in ("reorder", "grouping", "textgroups"):
        return list(range(n))
    return [i - spec.gold_index_base for i in R._flat_ints(R._gold_of(ex, spec)) if 0 <= i - spec.gold_index_base < n]


@cache
def _exposed_articles() -> frozenset[int]:
    """Outlier articles with at least one chunk that is a document of some eval outlier / fixedM / amzn row."""
    from .data import examples, row_key, sidecar
    labs = sidecar("outlier_articles.json.gz").get("rows") or {}
    ev = index()["outlier"]
    out = set()
    for fam in _OUTLIER:
        for split in ("train", "val"):
            for ex in examples(fam, split, decontaminate=False):
                for d, (a, _j) in zip(ex.get("documents") or [], labs.get(row_key(ex)) or []):
                    if a >= 0 and h64(norm(d.get("text"))) in ev:
                        out.add(a)
    return frozenset(out)


def assess(ex: dict, family: str) -> tuple[list[str], list[int]]:
    """(row-level reasons to DROP the row, positions of EVAL distractor items to scrub)."""
    from .data import row_key, sidecar
    docs = ex.get("documents") or []
    why: list[str] = []
    if int(row_key(ex), 16) & (2**64 - 1) in index()["rows"]:
        why.append("eval_row")
    if family == "textgroups":
        return why, []
    gp = set(_gold_positions(ex, family))
    if family in _ROW_QUESTION and any(is_eval(q, question=True) for q in ex.get("queries") or [] if isinstance(q, str)):
        why.append("eval_gold")
    flagged = [i for i, d in enumerate(docs) if is_eval(d.get("text"), question=d.get("type") == "query")]
    if gp & set(flagged) and "eval_gold" not in why:
        why.append("eval_gold")
    if family in _OUTLIER:
        labs = (sidecar("outlier_articles.json.gz").get("rows") or {}).get(row_key(ex)) or []
        exposed = _exposed_articles()
        if any(i < len(labs) and labs[i][0] in exposed for i in gp):
            why.append("eval_article")
    if flagged and family not in _SCRUB and not why:
        why.append("eval_item")
    return why, [i for i in flagged if i not in gp]


def _remap(x, old2new: dict[int, int], base: int):
    if isinstance(x, (list, tuple)):
        return [_remap(v, old2new, base) for v in x]
    return old2new[int(x) - base] + base


def scrub(ex: dict, family: str, drop: list[int]) -> dict:
    """``ex`` without the documents at ``drop`` (0-based, none of them gold); gold and hard-negative indices remapped."""
    from ctcbench_real import env as R

    from .data import spec_of
    spec = spec_of(family)
    base, gf = spec.gold_index_base, spec.extra.get("gold_field", "gold_doc_indices")
    dropset = set(drop)
    keep = [i for i in range(len(ex["documents"])) if i not in dropset]
    old2new = {o: n for n, o in enumerate(keep)}
    out = copy.deepcopy({k: v for k, v in ex.items() if k != "documents"})
    out["documents"] = [ex["documents"][i] for i in keep]
    out[gf] = _remap(R._gold_of(ex, spec), old2new, base)
    if ex.get("hard_neg_indices") is not None:
        out["hard_neg_indices"] = [old2new[i] for i in ex["hard_neg_indices"] if i in old2new]
    m = dict(out.get("_meta") or {}) if isinstance(out.get("_meta"), dict) else {}
    m["decontam_removed"] = len(drop)
    out["_meta"] = m
    return out


def filter_rows(family: str, exs) -> tuple[list[dict], collections.Counter]:
    """(cleaned rows, counts of reasons)."""
    kept, c = [], collections.Counter()
    for ex in exs:
        why, flagged = assess(ex, family)
        for x in why:
            c[x] += 1
        if why:
            c["dropped"] += 1
            continue
        if flagged:
            c["scrubbed"] += 1
            c["items_removed"] += len(flagged)
            ex = scrub(ex, family, flagged)
        kept.append(ex)
    c["kept"] = len(kept)
    return kept, c


__all__ = ["assess", "enabled", "filter_rows", "grams", "h64", "index", "is_eval", "norm", "scrub"]
