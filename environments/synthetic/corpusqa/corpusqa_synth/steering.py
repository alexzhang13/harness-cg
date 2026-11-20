"""Steering for corpusqa_synth: hints that describe / sketch the intended program, and program-level checks."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from synth_common import program as P

from .answers import entity_answer, periodic_answer
from .scoring import score_answer

STRATEGY_ENTITY = """\
`context` holds one long document per company (a quarterly or annual filing) or per school (a data set) -- from a
handful to several hundred documents of 100-300k characters each. Never send the whole corpus or the question to the
sub-LM, never send a whole filing, and do not read the documents yourself (code runs only inside ```repl blocks).
Stage 1 (one block): from the question, write a precise definition for each field you need (e.g. consolidated net income
including noncontrolling interests, current quarter; first-year vs transfer applicants; the first-year column). Locate,
in Python, only the sections those fields live in, in EVERY document. A financial statement is a title LINE (income /
operations, balance sheet, cash flows) followed within a few lines by its unit header ("(In millions, except per share
data)") and its own row labels with numbers; the table of contents, notes, MD&A and selected / quarterly data tables
also name the statements or print similar rows -- skip them. Do not write a regex per label: wording differs in every
filing. Print how many documents were located and which were not.
Stage 2 (one block): ONE llm_query_batched call, one prompt per document with its excerpt (smaller excerpts are read
more accurately), asking for a JSON object with "name" (from the cover page / title) and exactly your keys, money in
full units (or ask for the unit and convert: units differ per filer, and share counts can have their own unit). If the
sub-call budget is smaller than the number of documents (the error names it), put two documents per prompt and ask for
one JSON object per document. Parse every reply into a record; re-ask the documents with missing keys once, with a
narrower excerpt, in the same block; sanity-check (shares x EPS ~ net income).
Stage 3 (one block): compute the answer in Python over the records, format it exactly -- names as a JSON list
(json.dumps -- names contain commas), numbers with 2 decimals, percentages as xx.xx% -- and submit in the same block."""

STRATEGY_PERIODIC = (Path(__file__).parent / "hint_periodic.md").read_text().strip()

STRATEGY_MIXED = ("Identify the corpus first (from the first lines of a document, in your first block). If it is a series "
                  "of periodic releases:\n" + STRATEGY_PERIODIC + "\n\nIf it is one document per company or school:\n" +
                  STRATEGY_ENTITY)

ENTITY_BLOCKS = [r'''import re, json
FIELDS = {"<key>": "<the question's exact definition of this field>"}
NEED = ["cashflow"]       # the statements your fields live in: "income", "balance", "cashflow" (schools: CDS sections)
STMTS = {"income": (r"STATEMENTS? OF (?:INCOME|OPERATIONS|EARNINGS)", r"per share|income taxes"),
         "balance": (r"BALANCE SHEETS?", r"total assets|total liabilities|shareholders|stockholders"),
         "cashflow": (r"STATEMENTS? OF CASH FLOWS", r"operating activities")}
def locate(doc):          # cover + each needed statement: a title LINE, then a unit header and its own rows with numbers
    parts = []
    for title, rows in (STMTS[k] for k in NEED):
        for m in re.finditer(r"(?im)^[^\n]{0,40}" + title + r"[^\n]{0,40}$", doc):
            if re.search(r"\s\d{1,3}\s*$", m.group(0)): continue           # contents line ending in a page number
            after = doc[m.end(): m.end() + 2500]
            if (re.search(r"(?i)\((?:in|dollars in|\$ in|000s)[^)]{0,60}\)", "\n".join(after.split("\n")[:4]))
                    and re.search(r"(?i)" + rows, after) and len(re.findall(r"\d[\d,]*\d", after)) >= 8):
                parts.append(doc[max(0, m.start() - 200): m.end() + 4500]); break
    return doc[:1500] + "\n...\n" + "\n...\n".join(parts) if parts else None
excerpts = {i: locate(d) for i, d in enumerate(context)}
print(len(context), "documents;", sum(x is not None for x in excerpts.values()), "located; not located:",
      [i for i, x in excerpts.items() if x is None])''',
                 r'''spec = "\n".join(f'- "{k}": {v}' for k, v in FIELDS.items())
prompt = ('Return ONLY a JSON object with keys:\n- "name": the exact name on the cover page / title\n' + spec +
          "\nMoney in full units (multiply by the unit stated in the statement header); null if absent.\n\n")
def parse(r):
    m = re.search(r"\{.*\}", r or "", re.S)
    try: return json.loads(m.group(0)) if m else {}
    except ValueError: return {}
todo = [i for i, x in excerpts.items() if x]
rows = dict(zip(todo, map(parse, llm_query_batched([prompt + excerpts[i] for i in todo]))))  # > budget: 2 per prompt
bad = [i for i, j in rows.items() if any(j.get(k) is None for k in FIELDS)]
for i, r in zip(bad, llm_query_batched([prompt + excerpts[i] for i in bad])):    # re-ask ONCE
    j = parse(r)
    if sum(v is not None for v in j.values()) > sum(v is not None for v in rows[i].values()): rows[i] = j
print(len(rows), "records; still missing a key:", [i for i, j in rows.items() if any(j.get(k) is None for k in FIELDS)])''',
                 r'''recs = [j for j in rows.values() if all(j.get(k) is not None for k in FIELDS)]
# an inconsistent record (1000x off, sign against its label) is fixed by the unit / sign rule or left out -- not re-verified
answer["content"] = ...                                   # reduce over recs in Python, formatted exactly (json.dumps)
answer["ready"] = True''']


def _staged(blocks: list[str], kind: str = "") -> str:
    return "\n".join(f"# ---- {kind}stage {k + 1} (one block) ----\n" + b.strip() for k, b in enumerate(blocks))


PROGRAM_ENTITY = _staged(ENTITY_BLOCKS)
HINTS_PERIODIC = P.HintSpec(strategy=STRATEGY_PERIODIC)
HINTS_ENTITY = P.HintSpec(strategy=STRATEGY_ENTITY, program=PROGRAM_ENTITY)
HINTS_MIXED = P.HintSpec(strategy=STRATEGY_MIXED, program=_staged(ENTITY_BLOCKS, "one document per company / school: "))


def hints_for(family: str | None) -> P.HintSpec:
    return {"periodic": HINTS_PERIODIC, "entity": HINTS_ENTITY}.get(family or "", HINTS_MIXED)


def _st(s: dict) -> dict:
    return s.get("synth") or {}


def _hid(s: dict) -> dict:
    return s.get("synth_hidden") or {}


def _code(s: dict) -> str:
    return "\n".join(P.root_code(s))


FOR_CONTEXT = re.compile(r"for\s[^\n]*?\bin\s[^\n]*?\bcontext\b")
HEADLINE = re.compile(r"were at a seasonally adjusted|seasonally adjusted (annual rate|level) of|reported revenue of approx", re.I)


def _python_reads_all(s: dict) -> bool:
    return bool(FOR_CONTEXT.search(_code(s)))


def reports_read(s: dict) -> float:
    """Periodic: every release's table for the series was read, by sub-calls or by a Python loop over `context`."""
    h = _hid(s)
    if h.get("family") != "periodic":
        return 1.0
    req = h.get("required_ids") or []
    read = _st(s).get("read_ids") or {}
    got = [r for r in req if r in read]
    cov = len(got) / max(1, len(req))
    if cov >= 0.95:
        return 1.0
    if not got:
        return 1.0 if _python_reads_all(s) else 0.0
    dev = set()
    for r, rep in enumerate(h.get("reports") or []):
        for tid, t in rep["tables"].items():
            if t.get("layout") == "deviant":
                dev.add(f"R{r}:{tid}")
    if all(g in dev for g in got) and _python_reads_all(s):
        return 1.0
    return cov


def entities_read(s: dict) -> float:
    """Entity: every document's relevant sections were read (soft for mixed programs)."""
    h = _hid(s)
    if h.get("family") != "entity":
        return 1.0
    req = h.get("required_ids") or []
    read = _st(s).get("read_ids") or {}
    got = [r for r in req if r in read]
    if not got:
        return 1.0 if _python_reads_all(s) else 0.0
    return min(1.0, len(got) / max(1, len(req)) / 0.95)


def no_whole_corpus(s: dict) -> float:
    st = _st(s)
    return 1.0 if st.get("overlong", 0) == 0 and st.get("max_docs_per_call", 0) <= 8 else 0.0


def one_record_per_document(s: dict) -> float:
    return 1.0 if _st(s).get("max_docs_per_call", 0) <= 2 else 0.0


def no_guessed_requests(s: dict) -> float:
    return 1.0 if _st(s).get("guesses", 0) == 0 else 0.0


def _question_text(s: dict) -> str:
    info = s.get("info") or {}
    rp = (info.get("root_prompt") if isinstance(info, dict) else "") or ""
    return re.sub(r"\s+", " ", rp.split("Question: ", 1)[-1].split("\n\n`context`", 1)[0]).lower()


def no_summary_values(s: dict) -> float:
    if _st(s).get("summary_reads", 0) > 0:
        return 0.0
    code, q = _code(s), _question_text(s)
    for m in HEADLINE.finditer(code):
        tail = re.sub(r"\s+", " ", code[m.start(): m.end() + 25]).lower()
        if not q or tail not in q:
            return 0.0
    return 1.0


_YM = re.compile(r"^\d{4}[-/]\d{2}$")
_PCT = re.compile(r"^-?\d+\.\d{2}%$")
_NUM = re.compile(r"^-?\$?\d[\d,]*(\.\d+)?$")
_MONTH_TMPL = {"arg", "months_cmp", "topk_ratio", "months_mom", "prev2_avg", "cross_ma"}


def answer_format(s: dict) -> float:
    """The answer has the shape the output requirements ask for, where it affects the score (list, YYYY-MM, %, ...)."""
    fa = P.final_answer(s).strip()
    h = _hid(s)
    gold = h.get("gold") or {}
    kind, tmpl = gold.get("kind"), (h.get("spec") or {}).get("tmpl")
    if not fa:
        return 0.0
    if kind == "list":
        if not (fa.startswith("[") and fa.endswith("]")):
            if "," in fa or "\n" in fa:
                return 0.0
            if h.get("family") == "periodic" and tmpl in _MONTH_TMPL:
                return 1.0 if _YM.match(fa.strip().strip("\"'")) else 0.0
            return 1.0
        try:
            items = json.loads(fa)
        except json.JSONDecodeError:
            try:
                items = ast.literal_eval(fa)
            except (ValueError, SyntaxError):
                return 0.0
            if not isinstance(items, list) or any("," in str(x) for x in items):
                return 0.0
        if not isinstance(items, list):
            return 0.0
        items = [str(x) for x in items]
        if h.get("family") == "periodic" and tmpl in _MONTH_TMPL:
            return 1.0 if all(_YM.match(x.strip()) for x in items) else 0.0
        return 1.0
    if kind == "number":
        return 1.0 if _NUM.match(fa) else 0.0
    if isinstance(gold.get("value"), str) and gold["value"].endswith("%"):
        return 1.0 if _PCT.match(fa) else 0.0
    return 1.0 if len(fa.split()) <= 6 and not re.search(r"\d", fa) else 0.0


_UNIT_SENSITIVE = {"agg", "avg_filtered"}


def units_requested(s: dict) -> float:
    """Cross-filer money questions on mixed-unit corpora: every extraction sub-call asked for full units or the unit."""
    h = _hid(s)
    if h.get("domain") != "company":
        return 1.0
    spec = h.get("spec") or {}
    fields = spec.get("fields") or []
    cross = spec.get("tmpl") in _UNIT_SENSITIVE or (spec.get("tmpl") in ("arg", "topk") and len(fields) == 1)
    if not cross:
        return 1.0
    if len({e.get("dscale") for e in h.get("entities") or []}) < 2:
        return 1.0
    st = _st(s)
    n = st.get("company_extract_calls", 0)
    return 1.0 if n == 0 or st.get("unit_aware_calls", 0) >= n else 0.0


_SIGNATURE_TMPL = {"agg", "mom_abs_mean", "mean_ratio", "months_cmp", "topk_ratio", "months_mom",
                   "avg_filtered", "topk", "filter_cmp", "prev2_avg", "cross_ma", "filtered_mean", "topq_ratio"}


def _wrong_answer_check(alt_fn):
    def fn(s: dict) -> float:
        h = _hid(s)
        gold = h.get("gold") or {}
        if (h.get("spec") or {}).get("tmpl") not in _SIGNATURE_TMPL:
            return 1.0
        fa = P.final_answer(s)
        if not fa.strip():
            return 1.0
        try:
            alt = alt_fn(h)
        except Exception:  # noqa: BLE001  not applicable
            return 1.0
        if alt is None:
            return 1.0
        alt_txt = json.dumps(alt) if isinstance(alt, list) else str(alt)
        if score_answer(alt_txt, gold.get("value"), gold.get("kind")) >= 1.0:
            return 1.0
        hit = score_answer(fa, alt, gold.get("kind")) >= 1.0 and score_answer(fa, gold.get("value"), gold.get("kind")) < 1.0
        return 0.0 if hit else 1.0
    return fn


def _periodic_only(f):
    return lambda h: f(h) if h.get("family") == "periodic" else None


not_preliminary = _wrong_answer_check(_periodic_only(lambda h: periodic_answer(h, "own")))
not_stale = _wrong_answer_check(_periodic_only(lambda h: periodic_answer(h, "stale")))
not_other_adjustment = _wrong_answer_check(_periodic_only(lambda h: periodic_answer(h, swap_adj=True)))
not_unscaled = _wrong_answer_check(lambda h: periodic_answer(h, unscaled=True) if h.get("family") == "periodic"
                                   else (entity_answer(h, unscaled=True) if h.get("domain") == "company" else None))


def newest_release_rule(s: dict) -> float:
    if _hid(s).get("family") != "periodic":
        return 1.0
    code = _code(s)
    if re.search(r"\bmax\([^)]*key\s*=", code) and re.search(r"report|release|date", code, re.I):
        return 1.0
    has_sort = re.search(r"\bsorted\(|\.sort\(", code)
    first_wins = re.search(r"\.setdefault\(|\b(\w+)\s+not\s+in\s+(\w+)\s*:\s*\n?\s*\2\[\1\]\s*=", code)
    reverse = re.search(r"reverse\s*=\s*True", code)
    if has_sort and (not first_wins or reverse):
        return 1.0
    return 0.5 if re.search(r"latest|newest|most recent", code, re.I) else 0.0


_SCALE_SENSITIVE = {"agg", "mom_abs_mean", "count_thr", "filtered_mean"}


def unit_scale_applied(s: dict) -> float:
    h = _hid(s)
    if h.get("family") != "periodic" or (h.get("spec") or {}).get("tmpl") not in _SCALE_SENSITIVE:
        return 1.0
    return 1.0 if re.search(r"\*\s*\(?\s*(1_?000(_?000)?|1e[36]|10\s*\*\*\s*[36])\b|\bscale\b|thousand|million|\bUNIT\b",
                            _code(s), re.I) else 0.0


def period_filter(s: dict) -> float:
    if _hid(s).get("family") != "periodic":
        return 1.0
    return 1.0 if re.search(r"<=\s*[\w\[\]\"'.()]+\s*<=|\bperiod\b|\bLO\b|\bstart\w*\b[^\n]*\bend\w*\b", _code(s)) else 0.0


ENV_CHECKS = [
    P.Check("no_whole_corpus", no_whole_corpus, hard=True,
            doc="no sub-call over the window and none spanning more than 8 documents"),
    P.Check("one_record_per_doc", one_record_per_document, hard=True,
            doc="every sub-call reads sections of at most 2 documents (one record per document)"),
    P.Check("reports_read", reports_read, hard=True,
            doc="periodic: every release's table for the asked series was read (sub-LM or Python loop)"),
    P.no_aggregate_requests(hard=True),
    P.answer_not_copied(hard=True),
    P.Check("no_guessed_requests", no_guessed_requests, hard=True,
            doc="every sub-call named its series / field precisely enough that the sub-LM did not have to guess"),
    P.Check("no_summary_values", no_summary_values, hard=True,
            doc="no value taken from headline / MD&A prose (preliminary or rounded) instead of the tables"),
    P.Check("answer_format", answer_format, hard=True,
            doc="answer shape per the Output requirements: YYYY-MM months, bare number, xx.xx%, JSON list, bare name"),
    P.Check("units_requested", units_requested, hard=True,
            doc="mixed-unit filings: every extraction call asked for full units or for the unit"),
    P.Check("not_preliminary", not_preliminary, hard=True,
            doc="the answer is not the one each month's first (preliminary) print gives: earliest-release merge, "
                "own-month rows or headline sentences instead of the newest release"),
    P.Check("not_stale", not_stale, hard=True,
            doc="the answer is not the one a parser that skips revision-marked year rows gives (annual revision lost)"),
    P.Check("not_other_adjustment", not_other_adjustment, hard=True,
            doc="the answer is not the one the other seasonal adjustment's table gives"),
    P.Check("not_unscaled", not_unscaled, hard=True,
            doc="the answer is not the one values left in thousands / millions give"),
    P.Check("entities_read", entities_read, weight=0.5,
            doc="entity: every document's sections for the asked fields were read"),
    P.recovered_skips(),
    P.Check("newest_release_rule", newest_release_rule, weight=0.5,
            doc="code signature: releases merged in date order with the newest winning"),
    P.Check("unit_scale_applied", unit_scale_applied, weight=0.5,
            doc="code signature: table values multiplied by their unit"),
    P.Check("period_filter", period_filter, weight=0.5,
            doc="code signature: months restricted to the question's period"),
]

CHECKS = ENV_CHECKS
HARD = [c.name for c in ENV_CHECKS if c.hard]

__all__ = ["CHECKS", "ENV_CHECKS", "HARD", "HINTS_ENTITY", "HINTS_MIXED", "HINTS_PERIODIC", "hints_for"]
