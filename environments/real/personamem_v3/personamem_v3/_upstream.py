"""Vendored scoring + history-serialisation helpers from the official PersonaMem-v3 evaluation harness.

Source: https://github.com/bowen-upenn/PersonaMem-v3 (MIT-licensed code; the data is CC-BY-NC-4.0).
``strip_event`` / ``calendar_state`` come from ``evaluation/backend_query.py``, ``compact_event`` /
``calendar_entries`` from ``evaluation/inference_utils.py``, ``graded_ndcg_at_k`` from
``evaluation/tasks/personalized_recommendation.py``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


_LEAK_FIELDS_EVENT = ("split",)
_LEAK_FIELDS_PREF = (
    "split",
    "over_personalization_irrelevant",
    "update_history",
    "confidence_score_init",
    "confidence_cross_referenced",
    "stereotype_mark",
    "hidden_persona_labels",
)


def _strip_pref(pref: dict) -> dict:
    return {k: v for k, v in pref.items() if k not in _LEAK_FIELDS_PREF}


def strip_event(event: dict) -> dict:
    """``BackendQuery._strip_event``: drop scorer-side fields from an event."""
    out = {k: v for k, v in event.items() if k not in _LEAK_FIELDS_EVENT}
    if "preferences" in out:
        out["preferences"] = [_strip_pref(p) for p in out["preferences"]]
    return out


def compact_event(e: dict, strip_preferences: bool = False) -> dict:
    """``inference_utils._compact_event`` verbatim."""
    out = {
        "t": e.get("source_timestamp"),
        "when": e.get("formatted_timestamp"),
        "action": e.get("interaction_format", {}).get("action_label"),
        "user_message": e.get("interaction_format", {}).get("user_message"),
        "type": e.get("source_interaction_type"),
        "hashtags": e.get("source_hashtags", []),
        "content_type": e.get("content_type"),
        "author_id": e.get("author_id"),
        "is_self_authored": e.get("is_self_authored"),
        "is_dm": e.get("is_dm"),
    }
    if e.get("is_trending"):
        out["is_trending"] = True
        out["trending_topic"] = e.get("trending_topic")
        out["trending_relevance"] = e.get("trending_relevance")
    if e.get("is_ad"):
        out["is_ad"] = True
    loc = e.get("event_location")
    if isinstance(loc, dict) and loc.get("city"):
        out["location"] = {k: v for k, v in loc.items() if v is not None}
    content = e.get("content") or {}
    if "title" in content:
        out["title"] = content.get("title")
    if "caption" in content:
        out["caption"] = content.get("caption")
    if content.get("overall_description"):
        out["desc"] = content["overall_description"]
    if e.get("conversation"):
        out["conversation"] = e["conversation"]
    if not strip_preferences and e.get("preferences"):
        out["inferred_prefs"] = [
            {"persona_item": p.get("persona_item"), "category": p.get("category")}
            for p in e.get("preferences", [])
        ]
    return {k: v for k, v in out.items() if v not in (None, "", [])}


def calendar_state(modifications: Iterable[dict], as_of_timestamp: int) -> dict[str, dict]:
    """``BackendQuery.get_calendar_state``: fold the modification stream up to T."""
    state: dict[str, dict] = {}
    for m in modifications:
        if not isinstance(m, dict):
            continue
        ts = m.get("ts") or 0
        if ts > as_of_timestamp:
            continue
        action = m.get("action")
        if action == "added":
            entry = m.get("entry")
            if isinstance(entry, dict) and entry.get("entry_id"):
                state[entry["entry_id"]] = dict(entry)
        elif action == "updated":
            entry_id = m.get("entry_id")
            if entry_id and entry_id in state:
                diff = m.get("diff") or {}
                if isinstance(diff, dict):
                    for field, change in diff.items():
                        if isinstance(change, dict) and "to" in change:
                            state[entry_id][field] = change["to"]
        elif action == "removed":
            entry_id = m.get("entry_id")
            if entry_id and entry_id in state:
                state.pop(entry_id, None)
    return state


def calendar_entries(state: dict[str, dict]) -> list[dict]:
    """``inference_utils._render_calendar_block`` but returning structures
    instead of newline-joined JSON (the RLM harness offloads objects, not text).
    """
    out: list[dict] = []
    for _eid, entry in sorted((state or {}).items(),
                              key=lambda kv: (kv[1] or {}).get("start_ts", 0)):
        if not isinstance(entry, dict):
            continue
        loc = entry.get("location") or {}
        out.append({
            "title": entry.get("title", ""),
            "start": entry.get("formatted_timestamp") or entry.get("start_ts"),
            "end": entry.get("end_ts"),
            "location": loc.get("city", "") if isinstance(loc, dict) else loc,
            "attendees": entry.get("attendees"),
        })
    return out


NDCG_REL_TARGET = 2.0
NDCG_REL_FILLER = 1.0
NDCG_REL_HARD_NEG = -2.0


def graded_ndcg_at_k(
    ranked: list[int],
    positives: Iterable[int],
    hard_negs: Iterable[int],
    k: int,
    filler_grade: float = NDCG_REL_FILLER,
) -> float:
    """``personalized_recommendation._graded_ndcg_at_k`` verbatim."""
    pos = set(positives or [])
    hard = set(hard_negs or [])

    def _rel(idx: int) -> float:
        if idx in pos:
            return NDCG_REL_TARGET
        return NDCG_REL_HARD_NEG if idx in hard else filler_grade

    def _dcg(order: list[int]) -> float:
        return sum(_rel(idx) / math.log2(i + 2) for i, idx in enumerate(order[:k]))

    idcg = _dcg(sorted(ranked, key=_rel, reverse=True))
    if idcg <= 0:
        return 0.0
    return max(0.0, _dcg(ranked) / idcg)
