"""Steering for the CTC synthetic env: per-family hints (strategy / program tiers) and program-level checks."""

from __future__ import annotations

from synth_common import program as prog
from synth_common.program import Check, HintSpec

from . import programs as P
from .data import FAMILIES

def hints_for(fam: str) -> HintSpec:
    return HintSpec(strategy=P.STRATEGY[fam], program=P.program_text(fam))


HINTS = {f: hints_for(f) for f in FAMILIES}


def _fam(s: dict) -> str:
    return ((s.get("synth_hidden") or {}).get("fam")) or ""


def _a(s: dict, k: str) -> int:
    return int(prog.acct(s).get(k, 0) or 0)


def _signature(s: dict) -> float:
    """The family's intended program left its fingerprint in the sub-call log."""
    f = _fam(s)

    if f in ("nq", "hotpotqa"):
        return 1.0 if _a(s, "max_items_per_call") <= 60 else 0.0
    if f == "qdmatch_hpqa":
        return 1.0 if _a(s, "chain_calls") >= 1 and _a(s, "max_items_per_call") <= 60 else 0.0
    if f == "contradiction_iid":
        return 1.0 if _a(s, "pair_calls") >= 1 and _a(s, "set_search_large") == 0 else 0.0
    if f in ("outlier", "outlier_fixedM"):
        return 1.0 if _a(s, "same_article_calls") >= 1 and _a(s, "agg_requests") == 0 else 0.0
    if f == "grouping":
        return 1.0 if _a(s, "fixed_list_calls") >= 1 and _a(s, "agg_requests") == 0 else 0.0
    if f == "reorder":
        return 1.0 if _a(s, "link_calls") + _a(s, "score_calls") >= 1 and _a(s, "order_large") == 0 else 0.0
    if f == "textgroups":
        return 1.0 if _a(s, "passage_count_calls") == 0 else 0.0
    return 1.0


def _repl_outputs(s: dict) -> str:
    out = []
    for m in (s.get("rlm_history") or [])[2:]:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "user":
            out.append(str(m.get("content") if isinstance(m, dict) else getattr(m, "content", "")))
    return "\n".join(out)


def _coverage(s: dict) -> float:
    """Share of the required items that were READ: shown to a sub-call, or (retrieval families, where reading the
    hits yourself is a legitimate program) printed to the root."""
    hidden = s.get("synth_hidden") or {}
    req = hidden.get("required_ids") or []
    if not req:
        return 1.0
    read = prog.acct(s).get("read_ids") or {}
    printed = ""
    if hidden.get("fam") in ("nq", "hotpotqa"):
        printed = " ".join(_repl_outputs(s).split())
        texts = {str(i): t for i, t in hidden.get("items") or []}
    n = 0
    for r in req:
        if r in read:
            n += 1
        elif printed:
            head = " ".join((texts.get(str(r)) or "").split())[:80]
            n += bool(head) and head in printed
    return min(1.0, n / len(req) / 0.95)


def _json_replies(s: dict) -> float:
    calls = _a(s, "calls")
    return 1.0 if calls == 0 or _a(s, "json_prompts") >= 0.5 * calls else 0.0


SIGNATURE_DOC = {
    "nq": "documents read in pieces (sub-calls of <= 60 documents, or the root reading its own hits)",
    "hotpotqa": "documents read in pieces (sub-calls of <= 60 documents, or the root reading its own hits)",
    "qdmatch_hpqa": "a quoted fact-chain check, per-query sub-calls of <= 60 items",
    "contradiction_iid": "candidate pairs built in Python and labelled pair by pair; no 'find the pairs' search over "
                         "> 40 claims",
    "outlier": "same-article judgements on small groups; no 'which passages are outliers' aggregate",
    "outlier_fixedM": "same-article judgements on small groups; no 'which passages are outliers' aggregate",
    "grouping": "papers filed against a fixed category list; no 'cluster these papers' aggregate",
    "reorder": "local next/previous-passage judgements; no 'order these passages' request over > 16 passages",
    "textgroups": "values counted in Python, never sub-LM passage totals",
}

CHECKS = [
    Check("coverage", _coverage, hard=True, doc=">= 95% of the items a correct answer needs were read (by a sub-call; "
          "for nq/hotpotqa also by the root printing them)"),
    Check("no_whole_corpus_call", lambda s: 1.0 if _a(s, "whole_corpus_calls") == 0 else 0.0, hard=True,
          doc="no sub-call holding >= max(60 items, half the corpus)"),
    Check("signature", _signature, hard=True, doc="the family's program fingerprint (see SIGNATURE_DOC)"),
    prog.no_overlong(),
    Check("json_replies", _json_replies, doc=">= 50% of sub-calls ask for a fixed JSON reply shape"),
    prog.used_batched(),
    prog.answer_not_copied(),
]

__all__ = ["CHECKS", "HINTS", "SIGNATURE_DOC", "hints_for"]
