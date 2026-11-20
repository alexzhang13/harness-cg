"""CorpusOracle: the fake sub-LM for corpusqa_synth (contract: environments/synthetic/README.md)."""

from __future__ import annotations

import json
import re
import statistics
from collections import OrderedDict
from typing import Any

from synth_common.oracle import FakeSubLM, ReadModel, refuse_aggregate, rng_for, wants_json

from .entity import BOILER, CO_FIELDS, SCHOOL_BOILER
from .periodic import DOMAINS, FILLER, columns
from .scoring import OUTPUT_REQUIREMENTS
from .varied_oracle import CALIBRATION, VARIED_BUDGET_CHARS, VariedReader, is_varied

_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_WS = re.compile(r"\s+")
_VIEWS: OrderedDict[str, dict] = OrderedDict()
_STATIC_TEXT = list(FILLER) + list(BOILER) + list(SCHOOL_BOILER) + [
    "# EXPLANATORY NOTES", "FORWARD-LOOKING STATEMENTS", "Item 1A. Risk Factors", "# G. Annual Expenses and H. Financial Aid",
    OUTPUT_REQUIREMENTS]
_STATIC = {_WS.sub(" ", x).strip() for t in _STATIC_TEXT for x in _SPLIT.split(t) if x.strip()}
_STATIC_BLOB = "\n".join(sorted(_STATIC))
_META_LABEL = {"@name": "name", "@dscale": "unit", "@sscale": "share unit"}


def _doc_view(hidden: dict) -> dict:
    """All document text of this rollout as a segment set + a blob, to strip document text out of a prompt."""
    key = hidden.get("gen_key") or str(id(hidden["items"]))
    v = _VIEWS.get(key)
    if v is None:
        segs: set[str] = set()
        blobs = []
        for _, t in hidden["items"]:
            nt = _WS.sub(" ", t)
            blobs.append(nt)
            for piece in _SPLIT.split(t):
                s = _WS.sub(" ", piece).strip()
                if s:
                    segs.add(s)
        v = {"segs": segs, "blob": "\n".join(blobs)}
        _VIEWS[key] = v
        while len(_VIEWS) > 32:
            _VIEWS.popitem(last=False)
    else:
        _VIEWS.move_to_end(key)
    return v


_TVIEWS: OrderedDict[str, dict] = OrderedDict()
_MONTH_ROW = re.compile(r"^(?:<tr><td>|\|\s*)?(January|February|March|April|May|June|July|August|September|October|November|December)"
                        r"(?: \([rp]\))?(?:</td>|\s*\||\s{2,})")


def _table_view(hidden: dict) -> dict:
    """Per table: caption line and month rows, so the oracle can read the rows of a partial table."""
    key = hidden.get("gen_key") or str(id(hidden["items"]))
    v = _TVIEWS.get(key)
    if v is None:
        v = {}
        for iid, text in hidden["items"]:
            r, _, sec = iid[1:].partition(":")
            if not sec.startswith("T"):
                continue
            t = hidden["reports"][int(r)]["tables"][sec]
            lines = [ln for ln in text.split("\n") if ln.strip()]
            cap = next((_WS.sub(" ", ln).strip() for ln in lines if ln.startswith("Table ")), None)
            yms = list(t["rows"])
            month_lines = [_WS.sub(" ", ln).strip() for ln in lines if _MONTH_ROW.match(ln.strip())]
            rows = list(zip(month_lines, yms, strict=False)) if t["layout"] != "deviant" and len(month_lines) == len(yms) else []
            v[iid] = {"caption": cap, "rows": rows}
        seen: dict[str, int] = {}
        for tv in v.values():
            for ln, _ in tv["rows"]:
                seen[ln] = seen.get(ln, 0) + 1
        for tv in v.values():
            tv["uniq"] = [ln for ln, _ in tv["rows"] if seen[ln] == 1]
        _TVIEWS[key] = v
        while len(_TVIEWS) > 32:
            _TVIEWS.popitem(last=False)
    else:
        _TVIEWS.move_to_end(key)
    return v


def residual(prompt: str, hidden: dict) -> str:
    """The root's own words: the prompt minus every line/sentence that comes from a document (whole or cut)."""
    v = _doc_view(hidden)
    pieces = [(p.strip(), _WS.sub(" ", p).strip()) for p in _SPLIT.split(prompt)]
    pieces = [(raw, s) for raw, s in pieces if s]
    is_doc = [len(s) >= 24 and (s in v["segs"] or s in _STATIC) for _, s in pieces]
    keep = []
    for i, (raw, s) in enumerate(pieces):
        if s in v["segs"] or s in _STATIC:
            continue
        if len(s) >= 12 and ((i > 0 and is_doc[i - 1]) or (i + 1 < len(pieces) and is_doc[i + 1])) and (
                s in v["blob"] or s in _STATIC_BLOB):
            continue
        keep.append(raw)
    return "\n".join(keep)


def _norm(t: str) -> str:
    t = t.lower().replace("’", "'").replace("‘", "'").replace("&amp;", "&")
    t = re.sub(r"[-_/()\[\]{},:;\"`*]", " ", t)
    t = t.replace("'", "")
    return _WS.sub(" ", t)


def find_spans(text: str, table: dict[str, list[str]]) -> list[tuple[int, int, str]]:
    """Longest-first, non-overlapping word-boundary matches of synonyms. Returns (start, end, key) by position."""
    syn = sorted(((s, k) for k, ss in table.items() for s in ss), key=lambda x: -len(x[0]))
    taken = [False] * (len(text) + 1)
    out = []
    for s, k in syn:
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])", text):
            a, b = m.span()
            if any(taken[a:b]):
                continue
            for i in range(a, b):
                taken[i] = True
            out.append((a, b, k))
    return sorted(out)


EXTRACT_RE = re.compile(
    r"\b(extract|copy|transcribe|read off|list (the|all|every|each) (values?|figures?|numbers?|rows?|months?)|for (each|every) "
    r"(month|row|period|report|company|companies|university|universities|entity|document)|each month|every month|per month|"
    r"month by month|all (the )?months|one (line|row) per|json|keys?\s*:|return (the|a|only|exactly|all|its|their|these|each)|"
    r"report (the|its|their) values?|give (me )?the values?|values? (for|of) (each|every|all)|what (are|is) the values?|"
    r"provide (the|all|each)|fill in|look up|find the values?)\b", re.I)
AGG_RE = re.compile(
    r"\b(sum|summed|total of|in total|add up|average|mean|median|maximum|minimum|highest|lowest|largest|smallest|peak|"
    r"how many|count|which (month|year|region|company|companies|university|universities|one|ones)|what month|rank|"
    r"ranking|top \d+|top (three|five|ten)|greater than|exceed\w*|more than|less than|compare|difference|correlation|"
    r"percentage of (companies|universities|months))\b", re.I)
CONVERT_RE = re.compile(
    r"(full (us )?dollars|in (us )?dollars|actual (number|dollars|units|values?|shares)|full units|not in (thousands|millions)|"
    r"multiply|convert|absolute (number|dollars|units)|unscaled|raw (units|dollars)|in units\b|full (number|value)s?)", re.I)
PRIOR_RE = re.compile(r"\b(prior|previous|last year|year ago|year earlier|comparative|same (period|quarter) (in |of )?(the )?(prior|last|previous) year)\b")
KEYDEF_RE = re.compile(r"""^\s*(?:[-*•]\s*)?["'`]?([A-Za-z][A-Za-z0-9_]{0,40})["'`]?\s*(?::|=|->|—|–|\s-\s)\s*(.+)$""")

ADJ_SYN = {"NSA": ["not seasonally adjusted", "non seasonally adjusted", "unadjusted", "nsa", "not adjusted",
                   "without seasonal adjustment", "raw"],
           "SA": ["seasonally adjusted annual rate", "seasonally adjusted", "saar", "sa", "annual rate", "adjusted"]}
MEASURE_SYN = {
    "housing": {"permits": ["authorized by building permits", "authorised by building permits", "building permits",
                            "permits", "permit"],
                "authorized": ["authorized but not started", "authorized, but not started", "authorised but not started",
                               "not started"],
                "starts": ["housing starts", "starts", "started"], "completions": ["completions", "completed"],
                "under_construction": ["under construction"]},
    "retail": {"sales": ["sales"], "ecommerce": ["e commerce sales", "e commerce", "ecommerce", "online sales"],
               "inventories": ["inventories", "inventory"]},
    "jobs": {"openings": ["job openings", "openings", "vacancies"], "hires": ["hires", "hiring"], "quits": ["quits"],
             "layoffs": ["layoffs and discharges", "layoffs", "discharges"]},
}
COL_SYN = {
    "housing": {"g:United States": ["united states", "u s", "national", "nationwide", "nation", "country"],
                "g:Northeast": ["northeast", "north east"], "g:Midwest": ["midwest", "mid west"], "g:South": ["south"],
                "g:West": ["west"],
                "s:1 unit": ["single family", "1 unit", "one unit", "single unit"],
                "s:2 to 4 units": ["2 to 4 units", "2 to 4", "two to four units", "two to four", "2 4 units"],
                "s:5 units or more": ["5 units or more", "five units or more", "five or more units", "five or more unit",
                                      "five or more", "5 or more", "5+ units", "5+"],
                "s:Total": ["total"]},
    "retail": {"s:Retail and food services, total": ["retail and food services total", "total retail and food services",
                                                     "retail and food services", "total retail", "total"],
               "s:Motor vehicle and parts dealers": ["motor vehicle and parts dealers", "motor vehicle", "auto dealers"],
               "s:Food services and drinking places": ["food services and drinking places", "drinking places",
                                                       "restaurants", "food services"],
               "s:Nonstore retailers": ["nonstore retailers", "nonstore", "non store", "online retailers"],
               "s:General merchandise stores": ["general merchandise stores", "general merchandise"],
               "s:Gasoline stations": ["gasoline stations", "gasoline", "gas stations"]},
    "jobs": {"s:Total": ["total nonfarm", "nonfarm", "total"], "s:Construction": ["construction"],
             "s:Manufacturing": ["manufacturing"], "s:Retail trade": ["retail trade", "retail"],
             "s:Health care": ["health care", "healthcare"], "s:Northeast": ["northeast"], "s:South": ["south"],
             "s:Midwest": ["midwest"], "s:West": ["west"]},
}

CO_SYN = {
    "revenue": ["total revenues", "total revenue", "total net revenue", "net revenues", "net revenue", "net sales",
                "revenues", "revenue", "sales", "turnover"],
    "pretax": ["income before income taxes", "earnings before income taxes", "income loss before income taxes",
               "pre tax income", "pretax income", "income before taxes", "earnings before taxes", "pretax", "pre tax"],
    "tax": ["provision for income taxes", "income tax expense benefit", "income tax expense", "income tax provision",
            "tax expense", "income taxes", "income tax", "provision for taxes"],
    "net_income": ["consolidated net income", "net income loss", "net earnings loss", "net income", "net earnings",
                   "net loss", "net profit"],
    "eps_basic": ["basic earnings per share", "basic eps", "basic net income per share", "eps basic", "basic per share"],
    "eps_diluted": ["diluted earnings per share", "diluted eps", "diluted net income per share", "eps diluted",
                    "diluted per share"],
    "eps?": ["earnings per share", "eps", "per share"],
    "shares_basic": ["weighted average basic shares outstanding", "weighted average basic shares", "basic weighted average shares",
                     "basic shares", "shares basic", "basic weighted average", "weighted average shares basic"],
    "shares_diluted": ["weighted average diluted shares outstanding", "weighted average diluted shares",
                       "diluted weighted average shares", "diluted shares", "shares diluted", "diluted weighted average",
                       "weighted average shares diluted"],
    "shares?": ["weighted average shares outstanding", "weighted average shares", "shares outstanding", "share count"],
    "cfo": ["net cash provided by used in operating activities", "net cash provided by operating activities",
            "net cash from operating activities", "net cash used in operating activities", "cash flows from operating activities",
            "cash flow from operating activities", "cash from operating activities", "operating cash flow", "cash from operations",
            "operating activities", "cfo"],
    "cfi": ["net cash provided by used in investing activities", "net cash provided by investing activities",
            "net cash from investing activities", "net cash used in investing activities", "cash flows from investing activities",
            "cash flow from investing activities", "cash from investing activities", "investing cash flow", "investing activities",
            "cfi"],
    "cff": ["net cash provided by used in financing activities", "net cash provided by financing activities",
            "net cash from financing activities", "net cash used in financing activities", "cash flows from financing activities",
            "cash flow from financing activities", "cash from financing activities", "financing cash flow", "financing activities",
            "cff"],
    "cash_begin": ["cash and cash equivalents at beginning of period", "beginning of period", "beginning of the period",
                   "beginning cash", "cash at beginning"],
    "cash_end": ["cash and cash equivalents at end of period", "end of period", "end of the period", "ending cash", "cash at end"],
    "cash": ["cash and cash equivalents", "cash & cash equivalents", "cash and equivalents", "cash"],
    "total_assets": ["total assets", "assets"],
    "cur_debt": ["current portion of long term debt", "current maturities", "current portion"],
    "lt_debt": ["long term debt non current portion", "long term debt noncurrent", "non current portion of long term debt",
                "non current portion", "noncurrent portion", "long term debt", "long term borrowings", "debt"],
    "retained": ["retained earnings", "accumulated deficit", "reinvested earnings"],
    "total_equity": ["total equity", "equity including noncontrolling"],
    "equity": ["total shareholders equity", "total stockholders equity", "shareholders equity", "stockholders equity",
               "equity"],
    "da": ["depreciation and amortization", "depreciation depletion and amortization", "depreciation", "d&a"],
    "@name": ["company name", "registrant name", "registrant", "company", "name", "issuer"],
    "@dscale": ["dollar scale", "dollar unit", "unit of dollar amounts", "dollar amounts unit", "scale", "units", "unit"],
    "@sscale": ["share scale", "share unit", "unit of share counts", "shares unit", "share count unit", "share counts"],
}
SC_SYN = {
    "grand_total": ["grand total all students", "grand total enrollment", "grand total", "total enrollment",
                    "all students", "total students", "total student enrollment"],
    "ug_total": ["total all undergraduates", "total undergraduate students", "total undergraduates", "undergraduate students",
                 "undergraduate enrollment", "undergraduates", "undergraduate population"],
    "grad_total": ["total all graduate", "graduate and professional students", "graduate students", "graduate enrollment",
                   "graduate population"],
    "?applied": ["applied", "applicants", "applications"],
    "?admitted": ["admitted", "accepted", "admits"],
    "?enrolled": ["enrolled", "enrollees", "matriculated"],
    "grad_rate": ["six year graduation rate", "6 year graduation rate", "graduation rate"],
    "grad_rate_4": ["four year graduation rate", "4 year graduation rate"],
    "ft_faculty": ["full time instructional faculty", "full time faculty", "full time"],
    "pt_faculty": ["part time instructional faculty", "part time faculty", "part time"],
    "sections_100": ["class sections with 100 or more students", "sections with 100", "100 or more students",
                     "100 students or more", "100 or more", "100+", "large class sections", "class sections"],
    "pct_out_state": ["from out of state", "out of state"],
    "pct_housing": ["college owned", "college housing", "campus housing", "on campus", "live in college", "housing"],
    "@name": ["university name", "institution name", "university", "institution", "college", "school", "name"],
}


def _fmt_printed(v: float | None, dec: int) -> str:
    return "S" if v is None else f"{v:,.{dec}f}"


def _jnum(v: float | None, dec: int):
    if v is None:
        return None
    return int(round(v)) if dec == 0 else round(v, dec)


def make_index(items, min_len: int = 24, context_texts=None):
    """SegmentIndex without the residual set (this oracle computes it itself; halves the index cost)."""
    from synth_common.segindex import SegmentIndex

    return SegmentIndex(items, min_len=min_len)


class CorpusOracle(FakeSubLM):
    def __init__(self, read: ReadModel | None = None, share_slip: float = 0.12, calibration: dict | None = None,
                 varied_budget_chars: int = VARIED_BUDGET_CHARS):
        super().__init__(read or ReadModel(budget_chars=60_000, reliable_items=6, miss_rate=0.6, base_err=0.0005,
                                           load_err=0.08, agg_capacity=8, chatty=0.2, min_cov=0.6))
        self.share_slip = share_slip
        self.varied = VariedReader(self, calibration or CALIBRATION)
        self.varied_budget_chars = varied_budget_chars

    def __call__(self, prompt: str, state: Any) -> str:
        """Varied periodic corpora get the real sub-LM's window size."""
        hidden = state.get("synth_hidden") if isinstance(state, dict) else None
        if not is_varied(hidden) or len(str(prompt or "")) <= self.read.budget_chars:
            return super().__call__(prompt, state)
        budget = self.read.budget_chars
        try:
            self.read.budget_chars = self.varied_budget_chars
            return super().__call__(prompt, state)
        finally:
            self.read.budget_chars = budget

    def err_prob(self, prompt: str) -> float:
        ld = self.load(prompt)
        return self.read.base_err + self.read.load_err * ld * ld

    def respond(self, prompt: str, hidden: dict, st: dict) -> str:
        for k in ("guesses", "summary_reads", "multi_doc_calls", "max_docs_per_call", "unit_aware_calls",
                  "company_extract_calls"):
            st.setdefault(k, 0)
        self._st = st
        if is_varied(hidden):
            return self.varied.respond(prompt, residual(prompt, hidden), hidden, st)
        ids = self.given(prompt, hidden)
        instr = residual(prompt, hidden)
        if hidden["family"] == "periodic":
            return self._periodic(prompt, ids, instr, hidden, st)
        if not ids and not self._lines_present(prompt, hidden):
            return refuse_aggregate(st)
        return self._entity(prompt, ids, instr, hidden, st)

    @staticmethod
    def _count_docs(st: dict, docs: set) -> None:
        n = len(docs)
        st["max_docs_per_call"] = max(st.get("max_docs_per_call", 0), n)
        if n > 2:
            st["multi_doc_calls"] = st.get("multi_doc_calls", 0) + 1

    @staticmethod
    def _lines_present(prompt: str, hidden: dict) -> bool:
        lix = hidden.get("line_index") or {}
        return any(_WS.sub(" ", p).strip() in lix for p in _SPLIT.split(prompt))

    @staticmethod
    def _is_aggregate(instr: str) -> bool:
        return bool(AGG_RE.search(instr)) and not EXTRACT_RE.search(instr)

    def _periodic(self, prompt: str, ids: list[str], instr: str, hidden: dict, st: dict) -> str:
        dom_key = hidden["domain"]
        dom = DOMAINS[dom_key]
        cols = columns(dom)
        reports = hidden["reports"]
        tables = []
        for iid in ids:
            r, _, sec = iid[1:].partition(":")
            if sec.startswith("T"):
                tables.append((iid, int(r), sec))
        partial: dict[str, set] = {}
        pieces = {_WS.sub(" ", p).strip() for p in _SPLIT.split(prompt)}
        given = {t[0] for t in tables}
        for iid, tv in _table_view(hidden).items():
            if iid in given or not tv["caption"] or tv["caption"] not in pieces:
                continue
            got = {ym for ln, ym in tv["rows"] if ln in pieces}
            if got and any(ln in pieces for ln in tv["uniq"]):
                partial[iid] = got
                r, _, sec = iid[1:].partition(":")
                tables.append((iid, int(r), sec))
        self._partial = partial
        if not tables:
            heads = [int(iid[1:].partition(":")[0]) for iid in ids if iid.endswith(":head")]
            if heads:
                return self._periodic_head(prompt, instr, heads, hidden, st)
            if not ids:
                return refuse_aggregate(st)
            return self.chat(prompt, "The text you provided does not contain any statistical table — it is narrative "
                                     "text only, so I cannot read monthly values from it.")
        reqs = self._series_requests(instr, dom_key, cols)
        self._count_docs(st, {r for _, r, _ in tables})
        read, _skipped = self.read_items(prompt, [t[0] for t in tables], st)
        tables = [t for t in tables if t[0] in set(read)]
        if self._is_aggregate(instr):
            return self._periodic_aggregate(prompt, instr, tables, reqs, hidden, st)
        present = {(reports[r]["tables"][tid]["m"], reports[r]["tables"][tid]["adj"]) for _, r, tid in tables}
        blocks: list[tuple[str, dict]] = []
        missing = []
        by_report: OrderedDict[int, list] = OrderedDict()
        for iid, r, tid in tables:
            by_report.setdefault(r, []).append((iid, tid))
        for r, tl in by_report.items():
            rep = reports[r]
            out: dict[str, dict] = {}
            for label, spec in reqs:
                m, adj = self._resolve(spec, present, prompt, label)
                cand = [(iid, tid) for iid, tid in tl if rep["tables"][tid]["m"] == m and rep["tables"][tid]["adj"] == adj]
                if not cand:
                    missing.append(label or "the requested series")
                    continue
                iid, tid = cand[0]
                t = rep["tables"][tid]
                dec = hidden["decimals"][f"{m}:{adj}"]
                vals = {}
                yms = list(t["rows"])
                only = self._partial.get(iid)
                for j, ym in enumerate(yms):
                    if only is not None and ym not in only:
                        continue
                    ci = spec["col"]
                    v = t["rows"][ym][ci]
                    if v is not None and self.flip(prompt, f"{iid}:{ym}:{ci}:{label}"):
                        alt_c = [c for c in (ci - 1, ci + 1) if 0 <= c < len(cols) and t["rows"][ym][c] is not None]
                        rr = rng_for(prompt, iid, ym, "misread")
                        if alt_c and rr.random() < 0.6:
                            v = t["rows"][ym][rr.choice(alt_c)]
                        elif len(yms) > 1:
                            v = t["rows"][yms[j - 1 if j else 1]][ci]
                    vals[ym] = v
                out[label or self._col_label(cols, spec["col"])] = {"vals": vals, "dec": dec}
            blocks.append((f"{_month_name(rep['report'])} report", out))
        if not any(b[1] for b in blocks):
            what = ", ".join(sorted(set(missing))) or "the requested series"
            return self.chat(prompt, f"N/A — the provided text does not contain {what}.")
        return self._render_periodic(prompt, instr, blocks)

    def _periodic_head(self, prompt, instr, heads, hidden, st) -> str:
        """Answer from headline prose only (preliminary values, not the latest revision)."""
        dom_key = hidden["domain"]
        cols = columns(DOMAINS[dom_key])
        reqs = self._series_requests(instr, dom_key, cols)
        reports = hidden["reports"]
        self._count_docs(st, set(heads))
        read, _ = self.read_items(prompt, [f"R{r}:head" for r in heads], st)
        out = []
        for r in heads:
            if f"R{r}:head" not in read:
                continue
            facts = reports[r].get("head") or {}
            for label, spec in reqs:
                m = spec["m"] or (spec["ms"][0] if spec["ms"] else None)
                if m is None and len(facts) == 1:
                    m = next(iter(facts))
                f = facts.get(m) if m else None
                if not f or spec["adj"] == "NSA" or spec["col"] not in (0, 1):
                    continue
                vals = {ym: d.get(str(spec["col"])) for ym, d in f.items() if d.get(str(spec["col"])) is not None}
                if vals:
                    out.append((label, vals))
        if not out:
            return self.chat(prompt, "N/A — the summary text does not state that figure; it is only in the tables.")
        st["summary_reads"] += 1
        if wants_json(instr):
            obj = {}
            for label, vals in out:
                obj.update({ym: round(v) for ym, v in vals.items()} if label is None else {label: {ym: round(v) for ym, v in vals.items()}})
            return self.chat(prompt, json.dumps(obj))
        return self.chat(prompt, "\n".join(f"{ym}: {v:,.0f}" for _, vals in out for ym, v in vals.items()))

    def _col_label(self, cols, ci):
        g, s = cols[ci]
        return f"{g} - {s}"

    def _series_requests(self, instr: str, dom_key: str, cols) -> list[tuple[str | None, dict]]:
        reqs = []
        for line in instr.split("\n"):
            km = KEYDEF_RE.match(line)
            if not km:
                continue
            body = _norm(km.group(2))
            spec = self._series_spec(body, dom_key, cols)
            if spec["m"] or spec["adj"] or spec["col_said"]:
                for ci in spec["cols"]:
                    reqs.append((km.group(1), dict(spec, col=ci)))
        if reqs:
            return reqs
        spec = self._series_spec(_norm(instr), dom_key, cols)
        ms = spec["ms"] if len(spec["ms"]) > 1 else [spec["m"]]
        if len(spec["cols"]) > 1 or len(ms) > 1:
            out = []
            for m in ms:
                for ci in spec["cols"]:
                    lab = self._col_label(cols, ci) + (f" ({m})" if len(ms) > 1 else "")
                    out.append((lab, dict(spec, col=ci, m=m)))
            return out
        return [(None, dict(spec, col=spec["cols"][0]))]

    def _series_spec(self, text: str, dom_key: str, cols) -> dict:
        adj = None
        for _, _, k in find_spans(text, ADJ_SYN):
            adj = adj or k
        ms = []
        for _, _, k in find_spans(text, MEASURE_SYN[dom_key]):
            if k not in ms:
                ms.append(k)
        spans = find_spans(text, COL_SYN[dom_key])
        groups = [k[2:] for _, _, k in spans if k.startswith("g:")]
        subs = [k[2:] for _, _, k in spans if k.startswith("s:")]
        groups = list(dict.fromkeys(groups))
        subs = list(dict.fromkeys(subs))
        said = bool(groups or subs)
        idx = {c: i for i, c in enumerate(cols)}
        out_cols = []
        if dom_key == "housing":
            gs = groups or ["United States"]
            ss = [s for s in subs if s != "Total"] or ["Total"]
            if len(subs) > 1 and "Total" in subs and re.search(r"\b(to|and|versus|vs|with|over) (the )?total\b|\btotal (column|and)\b", text):
                ss = subs
            for g in gs:
                for s in ss:
                    if (g, s) in idx:
                        out_cols.append(idx[(g, s)])
                    elif (g, "Total") in idx and idx[(g, "Total")] not in out_cols:
                        out_cols.append(idx[(g, "Total")])
        else:
            for s in subs or [cols[0][1]]:
                for (_g, s2), i in idx.items():
                    if s2 == s and i not in out_cols:
                        out_cols.append(i)
        if not out_cols:
            out_cols = [0]
        return {"m": ms[0] if len(ms) == 1 else None, "ms": ms, "adj": adj, "cols": out_cols, "col_said": said}

    def _resolve(self, spec: dict, present: set, prompt: str, label) -> tuple[str, str]:
        """Pick (measure, adj) for a request among what the prompt contains; guess (seeded) when underspecified."""
        ms = [spec["m"]] if spec["m"] else (spec["ms"] or sorted({m for m, _ in present}))
        cand = [(m, a) for (m, a) in sorted(present) if m in ms and (spec["adj"] is None or a == spec["adj"])]
        if not cand:
            return (ms[0] if ms else "?", spec["adj"] or "SA")
        if len(cand) == 1:
            return cand[0]
        st = getattr(self, "_st", None)
        if st is not None:
            st["guesses"] = st.get("guesses", 0) + 1
        return rng_for(prompt, label, "ambiguous").choice(cand)

    def _render_periodic(self, prompt: str, instr: str, blocks) -> str:
        js = wants_json(instr)
        multi_rep = len(blocks) > 1
        if js:
            obj: dict[str, Any] = {}
            for rep_label, out in blocks:
                inner = {lab: {ym: _jnum(v, d["dec"]) for ym, v in d["vals"].items()} for lab, d in out.items()}
                if not multi_rep:
                    obj = next(iter(inner.values())) if len(inner) == 1 else inner
                else:
                    obj[rep_label] = inner if len(inner) > 1 else next(iter(inner.values()))
            body = json.dumps(obj)
            if rng_for(prompt, "fence").random() < self.read.chatty:
                body = "```json\n" + body + "\n```"
            return self.chat(prompt, body)
        lines = []
        for rep_label, out in blocks:
            if multi_rep:
                lines.append(f"{rep_label}:")
            for lab, d in out.items():
                if len(out) > 1:
                    lines.append(f"{lab}:")
                for ym, v in d["vals"].items():
                    lines.append(f"{ym}: {_fmt_printed(v, d['dec'])}")
        return self.chat(prompt, "\n".join(lines))

    def _periodic_aggregate(self, prompt, instr, tables, reqs, hidden, st) -> str:
        st["agg_requests"] += 1
        reports = hidden["reports"]
        label, spec = reqs[0]
        present = {(reports[r]["tables"][tid]["m"], reports[r]["tables"][tid]["adj"]) for _, r, tid in tables}
        m, adj = self._resolve(spec, present, prompt, label)
        vals: dict[str, float] = {}
        for _, r, tid in sorted(tables, key=lambda t: reports[t[1]]["report"]):
            t = reports[r]["tables"][tid]
            if t["m"] == m and t["adj"] == adj:
                for ym, row in t["rows"].items():
                    if row[spec["col"]] is not None:
                        vals[ym] = row[spec["col"]]
        if not vals:
            return self.chat(prompt, "N/A — the provided text does not contain that series.")
        rr = rng_for(prompt, "agg")
        exact = len(vals) <= self.read.agg_capacity
        low = instr.lower()
        if re.search(r"which month|what month|in which|when", low):
            pick = max(vals, key=vals.get) if re.search(r"high|large|max|peak|most", low) else min(vals, key=vals.get)
            if not exact:
                pick = rr.choice(sorted(vals, key=vals.get, reverse=bool(re.search(r"high|large|max|peak|most", low)))[:4])
            return self.chat(prompt, f"The month was {pick}.")
        if re.search(r"how many|count|number of", low):
            n = len(vals) // 2
            return self.chat(prompt, f"There are {max(0, round(n * rr.uniform(0.5, 1.5)))} such months.")
        xs = list(vals.values())
        if re.search(r"average|mean", low):
            x = statistics.mean(xs)
        elif "median" in low:
            x = statistics.median(xs)
        elif re.search(r"max|high|large|peak", low):
            x = max(xs)
        elif re.search(r"min|low|small", low):
            x = min(xs)
        else:
            x = sum(xs)
        if not exact:
            x *= rr.uniform(0.88, 1.12)
        return self.chat(prompt, f"The answer is approximately {x:,.1f}.")

    def _entity(self, prompt: str, ids: list[str], instr: str, hidden: dict, st: dict) -> str:
        dom = hidden["domain"]
        ents = hidden["entities"]
        secs: OrderedDict[int, set] = OrderedDict()
        for iid in ids:
            e, _, sec = iid[1:].partition(":")
            secs.setdefault(int(e), set()).add(sec)
        lix = hidden.get("line_index") or {}
        line_keys: dict[int, set] = {}
        for piece in _SPLIT.split(prompt):
            hit = lix.get(_WS.sub(" ", piece).strip())
            if hit:
                line_keys.setdefault(hit[0], set()).update(hit[1])
                secs.setdefault(hit[0], set())
        self._line_keys = line_keys
        self._count_docs(st, set(secs))
        read, _ = self.read_items(prompt, [f"E{e}" for e in secs], st)
        secs = OrderedDict((e, s) for e, s in secs.items() if f"E{e}" in set(read))
        reads = st.setdefault("read_ids", {})
        for e, have in secs.items():
            for sec in have:
                reads[f"E{e}:{sec}"] = reads.get(f"E{e}:{sec}", 0) + 1
        st["items_read"] = len(reads)
        reqs = self._field_requests(instr, dom)
        if self._is_aggregate(instr):
            return self._entity_aggregate(prompt, instr, secs, reqs, hidden, st)
        if not reqs:
            return self.chat(prompt, "Here is a summary: the document is a " +
                             ("quarterly report on Form 10-Q" if dom == "company" else "Common Data Set") +
                             ". Tell me which figures you need and I will extract them.")
        convert = bool(CONVERT_RE.search(instr))
        if dom == "company":
            st["company_extract_calls"] += 1
            if convert or any(f in ("@dscale", "@sscale") for _, f, _, _ in reqs):
                st["unit_aware_calls"] += 1
        out: OrderedDict[str, dict] = OrderedDict()
        for e, have in secs.items():
            ent = ents[e]
            row: dict[str, Any] = {}
            for key, field, per, amb in reqs:
                row[key] = self._lookup(prompt, ent, e, have, key, field, per, amb, convert, instr, dom)
            out[ent["legal"]] = row
        js = wants_json(instr)
        if js:
            obj = next(iter(out.values())) if len(out) == 1 else out
            body = json.dumps(obj)
            if rng_for(prompt, "fence").random() < self.read.chatty:
                body = "```json\n" + body + "\n```"
            return self.chat(prompt, body)
        lines = []
        for name, row in out.items():
            if len(out) > 1:
                lines.append(f"{name}:")
            for k, v in row.items():
                lines.append(f"{k}: {'N/A' if v is None else v}")
        return self.chat(prompt, "\n".join(lines))

    def _field_requests(self, instr: str, dom: str) -> list[tuple[str, str, str, bool]]:
        """[(output key, field, 'cur'|'prior', ambiguous?)]"""
        syn = CO_SYN if dom == "company" else SC_SYN
        reqs: list[tuple[str, str, str, bool]] = []
        seen = set()
        for line in instr.split("\n"):
            km = KEYDEF_RE.match(line)
            if not km:
                continue
            body = _norm(km.group(2))
            hit = self._first_field(body, syn, dom) or self._first_field(_norm(km.group(1)), syn, dom)
            if hit and km.group(1) not in seen:
                seen.add(km.group(1))
                field, amb = hit
                per = "prior" if (PRIOR_RE.search(body) or "prior" in km.group(1).lower()) else "cur"
                reqs.append((km.group(1), field, per, amb))
        if reqs:
            return reqs
        text = _norm(instr)
        for a, b, k in find_spans(text, syn):
            window = text[max(0, a - 50): b + 60]
            field, amb = self._disambiguate(k, window, dom)
            per = "prior" if PRIOR_RE.search(text[b: b + 60]) else "cur"
            key = _META_LABEL.get(field, field.rstrip("?")) if per == "cur" else f"{field}_prior"
            if key not in seen:
                seen.add(key)
                reqs.append((key, field, per, amb))
        return reqs

    def _first_field(self, body: str, syn, dom):
        sp = find_spans(body, syn)
        kinds = {k for _, _, k in sp}
        if dom == "company" and kinds & {"@dscale", "@sscale"} and re.search(r"\b(unit|scale|thousands|millions)\b", body):
            return ("@sscale" if re.search(r"\bshare", body) else "@dscale"), False
        real = [x for x in sp if not x[2].startswith("@")]
        pool = real or sp
        if not pool:
            return None
        a, b, k = max(pool, key=lambda x: (x[1] - x[0], -x[0]))
        return self._disambiguate(k, body, dom)

    def _disambiguate(self, k: str, window: str, dom: str) -> tuple[str, bool]:
        if dom == "company":
            if k == "net_income":
                if re.search(r"consolidated|including (non ?controlling|nci|minority)|before .{0,20}attributable|incl\.? nci", window):
                    return "net_income", False
                if "attributable" in window:
                    return ("ni_nci", False) if re.search(r"non ?controlling|minority|nci", window) else ("ni_company", False)
                return "net_income", True
            if k == "equity" and "total equity" not in window:
                return "equity", "shareholders" not in window and "stockholders" not in window
            if k == "eps?":
                return "eps?", True
            if k == "shares?":
                return "shares?", True
            return k, False
        if k.startswith("?"):
            base = k[1:]
            return ("tr_" if "transfer" in window else "fy_") + base, False
        if k == "pct_out_state" or k == "pct_housing":
            return (k + "_ug", False) if re.search(r"all (degree seeking )?undergraduates|all undergrad", window) else (k, False)
        return k, False

    def _lookup(self, prompt, ent, e, have, key, field, per, amb, convert, instr, dom):
        rr = rng_for(prompt, e, key, "lookup")
        lines = getattr(self, "_line_keys", {}).get(e, set())
        if field == "@name":
            known = dom == "school" or "cover" in have or bool(have & {"income", "balance", "cashflow"}) or ent["legal"].upper() in prompt
            return ent["legal"] if known else None
        if dom == "company":
            if field in ("@dscale", "@sscale"):
                if not have & {"income", "balance", "cashflow"}:
                    return None
                sc = ent["dscale"] if field == "@dscale" else ent["sscale"]
                if field == "@sscale" and ent["sscale"] != ent["dscale"] and rr.random() < self.share_slip:
                    sc = ent["dscale"]
                return int(sc) if re.search(r"1,?000|number", _norm(instr)) else {1e6: "millions", 1e3: "thousands", 1.0: "units"}[sc]
            guessed = field in ("eps?", "shares?") or (amb and field in ("net_income", "equity") and ent.get("has_nci"))
            if guessed:
                self._st["guesses"] = self._st.get("guesses", 0) + 1
            if field == "eps?":
                field = rr.choice(["eps_basic", "eps_diluted"])
            elif field == "shares?":
                field = rr.choice(["shares_basic", "shares_diluted"])
            elif amb and field == "net_income" and ent.get("has_nci"):
                field = rr.choice(["net_income", "ni_company"])
            elif amb and field == "equity" and ent.get("has_nci"):
                field = rr.choice(["equity", "total_equity"])
            if field == "cash_end":
                field = "cash"
            home = CO_FIELDS.get(field, {}).get("sec") or {"ni_company": "income", "ni_nci": "income", "cur_debt": "balance",
                                                          "total_equity": "balance", "cash_begin": "cashflow"}.get(field)
            pr = ent["printed"][per]
            if home not in have and field not in lines:
                if "mdna" in have and field in ent["prose"] and per == "cur":
                    self._st["summary_reads"] = self._st.get("summary_reads", 0) + 1
                    return ent["prose"][field]
                return None
            if field == "ni_company" and "ni_company" not in pr:
                field = "net_income"
            if field == "total_equity":
                v = pr.get("equity")
            elif field == "cash_begin":
                v = None
            else:
                v = pr.get(field)
            if v is None:
                return None
            if self.flip(prompt, f"{e}:{key}"):
                other = ent["printed"]["prior" if per == "cur" else "cur"].get(field)
                if other is not None:
                    v = other
            unit = CO_FIELDS.get(field, {}).get("unit", "usd")
            if convert and unit == "usd":
                v = v * ent["dscale"]
            elif convert and unit == "shares":
                sc = ent["sscale"]
                if sc != ent["dscale"] and rr.random() < self.share_slip:
                    sc = ent["dscale"]
                v = v * sc
            return round(v, 4) if isinstance(v, float) and v != int(v) else int(v)
        home_map = {"pct_out_state_ug": "F1", "pct_housing_ug": "F1", "grad_rate_4": "B4"}
        from .entity import SC_FIELDS
        home = SC_FIELDS.get(field, {}).get("sec") or home_map.get(field)
        if home not in have and field not in lines:
            return None
        v = ent["printed"]["cur"].get(field)
        if v is None:
            return None
        if self.flip(prompt, f"{e}:{key}"):
            v = v * rr.choice([0.9, 1.1]) if isinstance(v, int | float) else v
            v = int(round(v))
        return v

    def _entity_aggregate(self, prompt, instr, secs, reqs, hidden, st) -> str:
        """Confident and approximate: names a plausible subset / a number near a real one."""
        st["agg_requests"] += 1
        ents = hidden["entities"]
        rr = rng_for(prompt, "entity-agg")
        names = [ents[e]["legal"] for e in secs]
        low = instr.lower()
        if re.search(r"which|list|identify|rank|top|name", low):
            k = max(1, min(len(names), rr.randint(1, 3)))
            return self.chat(prompt, "Based on the documents: " + ", ".join(rr.sample(names, k)) + ".")
        if re.search(r"how many|count|number of", low):
            return self.chat(prompt, f"{rr.randint(0, len(names))}")
        vals = []
        for e in secs:
            for _key, field, per, _amb in reqs[:1]:
                v = ents[e]["printed"][per].get(field) if hidden["domain"] == "company" else ents[e]["printed"]["cur"].get(field)
                if isinstance(v, int | float):
                    vals.append(v)
        x = (statistics.mean(vals) if vals else rr.uniform(10, 1000)) * rr.uniform(0.7, 1.3)
        return self.chat(prompt, f"Approximately {x:,.2f}.")


def _month_name(ym: str) -> str:
    y, m = ym.split("-")
    return ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
            "November", "December"][int(m) - 1] + " " + y
