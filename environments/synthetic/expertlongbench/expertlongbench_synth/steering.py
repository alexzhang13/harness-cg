"""Steering for the synthetic ExpertLongBench env: hints in (`HINTS`), program-level checks out (`CHECKS`)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from synth_common.program import (
    Check,
    HintSpec,
    answer_not_copied,
    final_answer,
    max_turns,
    root_code,
)

from .env import NA_VALUES, field_score, is_na, parse_checklist, score
from .t01note import TURN1_C6, note_text, note_text_c6

STRATEGY = """\
How to solve this task:
- If `context` is longer than ~40,000 characters it will not fit one sub-call. Split it into ~40,000-character parts at
  line breaks and send ALL parts in one `llm_query_batched` call. Ask each part for every checklist field as JSON
  {"<field>": {"value": ..., "quote": ...}}: only what THAT part states, a short verbatim quote supporting each value,
  and "N/A" when the part does not state the field.
- Parse every reply robustly (JSON may be fenced or wrapped in chatter); re-ask any part whose reply does not parse on
  its two halves.
- Sub-LMs invent values for fields a part does not mention. In Python, keep a value only if its quote really occurs in
  that part (compare normalized words, not exact bytes).
- Merge per field across parts in Python: list fields are the union of all parts; when single values conflict, the
  latest-dated one wins (amended, extended, reassigned, successor) unless the field asks for the original (the filing
  date is the earliest). For a field whose verified candidates still disagree, send one small sub-call with ONLY those
  candidates and their quotes and ask for the final value.
- Submit `answer["content"] = json.dumps({field: value, ...})` with every field present. Write exactly "N/A" when
  nothing supports a field -- not "Not found", not "N/A (reason)" -- and never submit a Python dict.
- A short record (a few thousand characters) fits one sub-call or can be read directly; the answer format is the
  same."""

PROGRAM = """\
import re, json
FIELDS = [...]                                  # the checklist names, copied from the prompt

def split_chunks(text, size=40000):
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        k = text.rfind("\\n", i + size // 2, j) if j < len(text) else -1
        j = k if k > i else j
        out.append(text[i:j]); i = j
    return out

def ask(part, label):
    return (f"You are reading {label} of a case file. For each field report ONLY what this part states, "
            'as JSON {"<field>": {"value": "...", "quote": "<verbatim>"}}; use "N/A" if this part does not state it.\\n'
            "Fields:\\n" + "\\n".join(FIELDS) + f"\\n\\n=== {label} ===\\n{part}")

def parse_json(r):                              # first JSON object in a (possibly fenced / chatty) reply
    m = re.search(r"\\{.*\\}", r or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def quote_ok(q, part_norm):                      # most word 3-grams of the quote occur in the part
    w = norm(q).split()
    grams = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    return bool(grams) and sum(g in part_norm for g in grams) >= 0.6 * len(grams)

parts = split_chunks(context)
replies = llm_query_batched([ask(p, f"PART {i + 1} OF {len(parts)}") for i, p in enumerate(parts)])
records = [(p, parse_json(r)) for p, r in zip(parts, replies)]
# ... re-ask parts whose reply is None on their halves ...
cands = {f: [] for f in FIELDS}
for p, obj in records:
    pn = norm(p)
    for f in FIELDS:
        v = (obj or {}).get(f) or {}
        val, q = (v.get("value"), v.get("quote")) if isinstance(v, dict) else (v, "")
        if val and str(val).strip().upper() != "N/A" and quote_ok(q, pn):
            cands[f].append((str(val), str(q)))
final = {}
for f in FIELDS:
    vals = list(dict.fromkeys(v for v, _ in cands[f]))
    final[f] = vals[0] if len(vals) == 1 else ("N/A" if not vals else None)
need = [f for f in FIELDS if final[f] is None]  # conflicts / lists: union, latest-dated wins
outs = llm_query_batched([f"Field: {f}\\nCandidates with quotes:\\n" + "\\n".join(f"- {v} | {q}" for v, q in cands[f])
                          + "\\nGive the final value for the whole case (lists: all distinct items; conflicts: the "
                          "latest-dated). Return only the value." for f in need])
for f, o in zip(need, outs):
    final[f] = o.strip() or "N/A"
answer["content"] = json.dumps(final)
answer["ready"] = True"""

@dataclass
class DocketHintSpec(HintSpec):
    """HintSpec plus tier ``note`` (the T01 note in t01note.py)."""
    note: str = ""
    note_v4: str = ""
    note_v5: str = ""

    def text(self, tier: str) -> str:
        if tier == "note":
            return self.note.strip()
        if tier == "note_v4":
            return (self.note_v4 or note_text("v4")).strip()
        if tier == "note_v5":
            return (self.note_v5 or note_text("v5")).strip()
        if tier == "note_c6":
            return note_text_c6().strip()
        return super().text(tier)


HINTS = DocketHintSpec(note=note_text_c6(), note_v4=note_text("v4"), note_v5=note_text("v5"), strategy=STRATEGY,
                       program=PROGRAM)


def _hidden(s: dict) -> dict:
    return s.get("synth_hidden") or {}


def _docket(s: dict) -> bool:
    return _hidden(s).get("variant") == "docket"


def _acct(s: dict) -> dict:
    return s.get("synth") or {}


def _answer_dict(s: dict) -> dict:
    fa = final_answer(s)
    try:
        obj = json.loads(fa)
        if isinstance(obj, dict):
            return {str(k).strip().lower(): ("" if v is None else str(v)) for k, v in obj.items()}
    except Exception:  # noqa: BLE001
        pass
    return {k.strip().lower(): v for k, v in parse_checklist(fa).items()}


def _coverage(s: dict) -> float:
    if not _docket(s):
        return 1.0
    true_ids = [str(f["id"]) for f in _hidden(s).get("facts") or [] if f["kind"] == "true"]
    if not true_ids:
        return 1.0
    ev = _acct(s).get("evidence_read") or {}
    cov = sum(1 for i in true_ids if i in ev) / len(true_ids)
    return min(1.0, cov / 0.95)


def _json_answer(s: dict) -> float:
    try:
        obj = json.loads(final_answer(s))
    except Exception:  # noqa: BLE001
        return 0.0
    if not isinstance(obj, dict):
        return 0.0
    keys = {str(k).strip().lower() for k in obj}
    return 1.0 if all(f["name"].strip().lower() in keys for f in _hidden(s).get("fields") or []) else 0.0


_DECLINE = re.compile(r"^\s*(?:n/?a\b.*\S|not (?:found|stated|specified|provided|available|mentioned\b.*\S|applicable\b.*\S|"
                      r"discussed|addressed|given|indicated|listed|identified)|unknown|unclear|none (?:found|stated|"
                      r"mentioned|listed|identified|given)|no (?:information|data|mention)|cannot be determined|"
                      r"unable to)", re.I)


def _na_token(s: dict) -> float:
    ans = _answer_dict(s)
    if not ans:
        return 0.0
    bad = [v for v in ans.values() if not is_na(v) and _DECLINE.match(str(v))]
    return 0.0 if bad else 1.0


def _no_whole_file(s: dict) -> float:
    if not _docket(s):
        return 1.0
    n = int(_hidden(s).get("context_chars") or 0)
    if not n:
        return 1.0
    big = [e for e in (s.get("rlm_sub_llm_log") or []) if int(e.get("prompt_chars") or 0) >= 0.9 * n]
    return 0.0 if big else 1.0


def _contaminated(s: dict, log_key: str) -> float:
    """1.0 if no logged confabulation appears in the answer. ``log_key="stale"`` checks outdated latest-wins values."""
    if log_key == "stale":
        log = [[f["key"], f["val"]] for f in _hidden(s).get("facts") or [] if f["kind"] == "stale"]
    else:
        log = _acct(s).get(log_key) or []
    if not log:
        return 1.0
    hidden = _hidden(s)
    by_key = {f["key"]: f for f in hidden.get("fields") or []}
    ans = _answer_dict(s)
    for key, val in log:
        f = by_key.get(key)
        if f is None:
            continue
        if log_key == "fabricated" and f["type"] in ("yesno", "enum"):
            continue
        cand = ans.get(f["name"].strip().lower(), "")
        if is_na(cand):
            continue
        gold = f["gold"]
        items = val if isinstance(val, list) and f["type"] in ("dates", "casenos", "cites", "statutes", "remedies",
                                                               "orgs", "entities", "persons", "words") else [val]
        for it in items:
            if isinstance(gold, list) and it in gold or it == gold:
                continue
            probe = dict(f)
            probe["gold"] = [it] if isinstance(val, list) and f["type"] not in ("person",) else it
            try:
                if field_score(probe, cand, hidden) > 0:
                    return 0.0
            except Exception:  # noqa: BLE001
                continue
    return 1.0


_QUOTE = re.compile(r"quote", re.I)
_VERIFY = re.compile(r"quote_ok|verif\w*|support\w*\s*\(|\bin\s+\w*(?:part|chunk|src|source|text|ctx|context|norm|doc)"
                     r"\w*|\.find\(|\.count\(|grams?\b", re.I)


def _quote_check(s: dict) -> float:
    if not _docket(s):
        return 1.0
    return 1.0 if any(_QUOTE.search(c) and _VERIFY.search(c) for c in root_code(s)) else 0.0


def _merge(s: dict) -> float:
    hidden = _hidden(s)
    scores = s.get("elb_field_scores")
    if scores is None:
        tmp: dict = {}
        score(final_answer(s), hidden.get("gold"), hidden, tmp)
        scores = tmp.get("elb_field_scores") or {}
    stale = {f["key"] for f in hidden.get("facts") or [] if f["kind"] == "stale"}
    sens = [f["name"] for f in hidden.get("fields") or []
            if (f["rule"] == "union" and isinstance(f["gold"], list) and len(f["gold"]) >= 2)
            or (f["rule"] == "latest" and f["key"] in stale and f["gold"] != "N/A")]
    if not sens:
        return 1.0
    return sum(float(scores.get(n, 0.0)) for n in sens) / len(sens)


def _batched(s: dict) -> float:
    if not _docket(s):
        return 1.0
    return 1.0 if _acct(s).get("batches", 0) >= 1 else 0.0


def _call_budget(s: dict) -> float:
    n = int(_hidden(s).get("context_chars") or 0)
    parts = max(1, math.ceil(n / 40_000))
    limit = 2 * parts + len(_hidden(s).get("fields") or []) + 10
    return 1.0 if int(_acct(s).get("calls", 0)) <= limit else 0.0


_HDR_PARSE = re.compile(r"DOCKET FOR CASE|Date Filed:|Assigned to:", re.I)
_REDUCE = re.compile(r"reduce|finaliz\w*|final value", re.I)


def _hint_text(s: dict) -> str:
    for m in (s.get("rlm_history") or [])[:2]:
        c = str((m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or "")
        if "<hint>" in c:
            return c.split("<hint>", 1)[1].split("</hint>", 1)[0]
    return ""


def note_version(s: dict) -> str:
    """Docket note version the row carried ("v4" / "v5" / "c6"); unhinted rows count as c6."""
    c = _hint_text(s)
    if "def plaintiff_blocks(" in c:
        return "v4"
    if "def plaintiff_counsel(" in c:
        return "v5"
    return "c6"


_C6_LOOK = re.compile(r"print\(\s*context\b|print\(\s*context\s*\[|\bcontext\.(?:find|count|index)\(|"
                      r"re\.(?:findall|search|finditer)\([^\n]*,\s*context\b")


def _root_turns(s: dict) -> list[str]:
    return [str((m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or "")
            for m in (s.get("rlm_history") or [])[2:]
            if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "assistant"]


def follows_c6_parts(s: dict) -> dict[str, bool]:
    """Did the trace follow note c6 (probe, one batched fan-out, batched reduce, <= 4 turns, no direct reads)?"""
    turns = _root_turns(s)
    code = "\n".join(turns)
    a = _acct(s)
    own = {ln.strip() for ln in TURN1_C6.splitlines() if ln.strip()}
    looks = [ln for ln in code.splitlines() if ln.strip() not in own and _C6_LOOK.search(ln)]
    return {"header_parse": bool(_HDR_PARSE.search(code)),
            "probe": len(a.get("overviews") or []) >= 1,
            "fields_defs": bool(re.search(r"\bFIELDS\s*=\s*\{", code)),
            "fanout": a.get("batches", 0) >= 1 and "llm_query_batched" in code and a.get("max_batch", 0) >= 4,
            "reduce": a.get("batches", 0) >= 2,
            "json_dumps": "json.dumps" in code,
            "no_look": not looks,
            "turns_le_4": 0 < int(s.get("rlm_iterations") or 0) <= 4}


def follows_note_parts(s: dict) -> dict[str, bool]:
    """Did the trace follow the v4 / v5 T01 note (one batched fan-out, quote check, batched reduce, <= 6 turns)?"""
    code = "\n".join(str((m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or "")
                     for m in (s.get("rlm_history") or [])[2:]
                     if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "assistant")
    a = _acct(s)
    out = {"header_parse": bool(_HDR_PARSE.search(code)),
           "fanout": a.get("batches", 0) >= 1 and "llm_query_batched" in code and a.get("max_batch", 0) >= 2,
           "quote_check": bool(_QUOTE.search(code) and _VERIFY.search(code)),
           "reduce": bool(_REDUCE.search(code)) and a.get("batches", 0) >= 2,
           "role_map": bool(re.search(r"\bROLE\s*=\s*\{", code)),
           "json_dumps": "json.dumps" in code,
           "turns_le_6": 0 < int(s.get("rlm_iterations") or 0) <= 6}
    if note_version(s) == "v5":
        out["counsel_listing"] = "plaintiff_counsel(" in code
        out["outcome_merge"] = bool(re.search(r"\bOUTCOME\s*=", code)) and bool(
            re.search(r"[\"']settlement_date[\"']\s*:", code))
    return out


def _follows_note(s: dict) -> float | None:
    if not _docket(s):
        return None
    parts = follows_c6_parts(s) if note_version(s) == "c6" else follows_note_parts(s)
    return 1.0 if all(parts.values()) else 0.0


CHECKS = [
    Check("coverage", _coverage, hard=True, weight=2.0,
          doc=">= 95% of the true evidence sentences reached a sub-call unskipped (docket rows)"),
    Check("json_answer", _json_answer, hard=True, weight=2.0,
          doc="the answer is a JSON string with every checklist field"),
    Check("na_token", _na_token, hard=True, weight=2.0,
          doc="declines use the grader's N/A token, never 'Not found' / 'N/A (reason)'"),
    Check("no_whole_file", _no_whole_file, hard=True, weight=2.0, doc="no sub-call carries >= 90% of the docket"),
    Check("not_over_delegated", lambda s: _contaminated(s, "fabricated"), hard=True, weight=2.0,
          doc="no value the sub-LM fabricated (for a part that lacked the field) reached the answer"),
    Check("latest_wins", lambda s: _contaminated(s, "stale"), hard=True, weight=2.0,
          doc="no superseded value (original fund, first monitor, pre-extension term/deadline, first judge) submitted"),
    Check("quote_check", _quote_check, doc="the root's code checks sub-LM quotes against the source"),
    Check("merge", _merge, doc="merge-sensitive fields right: lists unioned across parts, latest value wins"),
    Check("no_distractor_values", lambda s: _contaminated(s, "distractor_picks"),
          doc="no distractor value (magistrate, nominee, demand amount, ...) reached the answer"),
    Check("batched", _batched, doc="docket parts fanned out with llm_query_batched"),
    max_turns(20),
    answer_not_copied(min_len=20),
    Check("call_budget", _call_budget, doc="sub-calls <= 2 x parts + fields + 10 (no verify loops)"),
    Check("follows_note", _follows_note, weight=0.0,
          doc="trace adherence to the row's T01 note (docket rows; monitoring only, weight 0). c6 (final): header "
              "regexes + probe sub-call + FIELDS + ONE batched fan-out over >= 4 parts + batched reduce + json.dumps, "
              "<= 4 turns, no root look at the file; v4 / v5: header parse + fan-out + quote check + ROLE reduce, "
              "<= 6 turns"),
]

JUDGE_CRITERIA = [
    "Chunks the whole file (or reads a short record whole) instead of sampling a prefix or a few grep hits.",
    "Asks each sub-call only about the text it is given, with field names copied from the checklist.",
    "Treats sub-LM output as claims: checks quotes/evidence against the source before using a value.",
    "Merges in Python per field: union for lists, latest-dated for amended values, earliest for the original filing.",
    "Sends conflicts back to the sub-LM with only the conflicting candidates and their evidence.",
    "Submits json.dumps of a dict with every field and the bare N/A token; no prose declines, no dict repr.",
    "Spends turns on the plan, not on printing consecutive slices of the file.",
]

__all__ = ["CHECKS", "HINTS", "JUDGE_CRITERIA", "NA_VALUES", "PROGRAM", "STRATEGY"]
