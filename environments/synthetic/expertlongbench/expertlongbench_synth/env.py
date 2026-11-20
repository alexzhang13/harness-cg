"""Synthetic ExpertLongBench: fill an expert checklist from a long case file (or a short clinic dialogue)."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Any

from synth_common import FakeSubLM, ReadModel, refuse_aggregate, rng_for
from synth_common.oracle import wants_json
from synth_common.segindex import _SPLIT, _WS

from .generate import CLINIC_FIELDS, CONDUCT, DOCKET_FIELDS, REMEDIES, STATUTES, TOPICS, generate
from .textkit import (
    ORDINAL,
    cite_key,
    find_amounts,
    find_casenos,
    find_cites,
    find_dates,
    find_durations,
    fmt_amount,
    fmt_caseno,
    fmt_cite,
    fmt_date,
    fmt_duration,
    fmt_person,
    norm,
    person_match,
    token_overlap,
)

VALUE_READ_TYPES = ("events", "cause", "terms", "counsel_type", "parties")
NA_VALUES = {"n/a", "na", "none", "not applicable", "not mentioned", ""}
ALL_FIELDS = {**DOCKET_FIELDS, **CLINIC_FIELDS}


def is_na(v: Any) -> bool:
    return str(v if v is not None else "").strip().lower() in NA_VALUES


_FACT_CACHE: OrderedDict = OrderedDict()

TRAP_RATE = {
    "hearing": {"excl": 0.08, "none": 0.45, "broad": 0.9},
    "until_further": {"excl": 0.08, "none": 0.6},
    "denial": {"excl": 0.08, "none": 0.6},
    "defense": {"excl": 0.04, "none": 0.3},
    "transfer": {"excl": 0.12, "none": 0.35},
    "cited": {"excl": 0.1, "none": 0.4},
    "gov_defense": {"excl": 0.04, "none": 0.3, "broad": 0.5},
    "names": {"excl": 0.08, "none": 0.5},
    "routine": {"excl": 0.05, "none": 0.25},
}
REDUCE_KEEP = {"defense": {"excl": 0.04, "none": 0.5}, "cited": {"excl": 0.1, "none": 0.6},
               "gov_defense": {"excl": 0.04, "none": 0.5, "broad": 0.6}, "names": {"excl": 0.05, "none": 0.6},
               "routine": {"excl": 0.05, "none": 0.3}}
OVERRIDE_RATE = {"cert_denied": {"excl": 0.08, "none": 0.45, "broad": 0.85},
                 "signed": {"excl": 0.05, "none": 0.35}}
TRAP_DEF = {"gov_defense": "counsel_gov", "names": "parties", "routine": "filings", "cert_denied": "class_action",
            "signed": "settlement_date"}
ROUTINE_CAP = 2
P_MISS_APPROVAL = 0.45
P_NARROW_ENFORCED = 0.6
FAB_SKIP_OK = 0.1
P_TRUST_LISTING = 0.6
_NEG_NEAR = r"(?:not|never|no|exclud\w*|except)"


def field_segs(low: str, fields: list[dict]) -> dict[str, str]:
    """Each requested field's own definition: the prompt text from the field's name to the next field's name."""
    pos = {}
    for f in fields:
        n = f["name"].lower()
        for pat in (f'"{n}"', f"'{n}'", f"- {n}", n):
            i = low.find(pat)
            if i != -1:
                pos[f["name"]] = (i, len(pat))
                break
    order = sorted(pos.items(), key=lambda kv: kv[1][0])
    segs = {}
    for j, (name, (i, ln)) in enumerate(order):
        end = order[j + 1][1][0] if j + 1 < len(order) else min(len(low), i + 600)
        segs[name] = low[i + ln:min(end, i + ln + 600)]
    return segs


def def_mode(key: str, seg: str) -> str:
    """How a field's definition treats that field's convention trap: excl | broad | none (| no_mag for the judge)."""
    seg = seg or ""
    if key == "trial":
        hear = r"hearings? on (?:major )?motions|motion hearings?|oral arguments?"
        clauses = [c for c in re.split(r"[.;(]", seg) if re.search(hear, c)]
        if clauses and not any(re.search(r"\b" + _NEG_NEAR + r"\b", c) for c in clauses):
            return "broad"
        if clauses:
            return "excl"
        if re.search(r"actually held|evidentiary hearing|not (?:a )?(?:motion )?hearing", seg):
            return "excl"
    elif key == "decree_duration":
        if "until further order" in seg:
            return "excl"
    elif key == "decree_terms":
        if re.search(r"deni\w*[^.;]{0,60}(?:no terms|n/a|imposes no)|(?:no terms|imposes no)[^.;]{0,60}den|"
                     r"denial imposes no|only den\w*[^.;]{0,30}n/a", seg):
            return "excl"
    elif key == "counsel_type":
        if re.search(r"never the defendants|not the defendants|plaintiffs'? (?:lawyers |counsel |side )?only|"
                     r"only the plaintiffs|defense (?:firms|counsel|lawyers)[^.;]{0,20}(?:not|never|exclud)", seg):
            return "excl"
    elif key == "related":
        if re.search(r"docket sheet|merely cited|not (?:case numbers |cases )?(?:merely )?cited|never[^.;]{0,30}cited|"
                     r"precedents?[^.;]{0,20}(?:not|never)|header", seg):
            return "excl"
    elif key == "filing_date":
        if re.search(r"earliest", seg):
            return "excl"
    elif key == "judge":
        if re.search(_NEG_NEAR + r"[^.;]{0,15}magistrate", seg) and not re.search(
                r"magistrate judge only when|magistrate[^.;]{0,30}only (?:when|if)|unless[^.;]{0,40}assigned", seg):
            return "no_mag"
    elif key == "class_action":
        if re.search(r"even if certification|never granted|regardless of certification", seg):
            return "excl"
        if re.search(r"not denied certification|and not denied|denied certification", seg):
            return "broad"
    elif key == "class_desc":
        return "describe" if re.search(r"described as|how many|what kind", seg) else "none"
    elif key == "parties":
        if re.search(r"do not list|without listing|not list every|describe", seg):
            return "excl"
    elif key == "counsel_gov":
        if re.search(r"only when the government|count only when|government[^.;]{0,80}only (?:when|if)", seg):
            return "excl"
        if "government" in seg:
            return "broad"
    elif key == "counsel_prose":
        return "excl" if "later represented" in seg else "none"
    elif key == "settlement_date":
        if re.search(r"approv", seg):
            return "excl"
    elif key == "court_enforced":
        return "inj" if re.search(r"injunction|judgment|court-ordered relief", seg) else "none"
    elif key == "filings":
        if re.search(r"not routine|routine briefing|responses|status reports", seg):
            return "excl"
    elif key == "filings_outcome":
        return "outcome" if "outcome" in seg else "none"
    return "none"


def _grams(text: str, n: int = 4) -> list[str]:
    w = norm(text).split()
    return [" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))]


class CaseFileOracle(FakeSubLM):
    """Checklist-field extractor over the evidence sentences present in the prompt."""

    def __init__(self, read: ReadModel | None = None, reliable_frac: float = 0.5, miss_rate: float = 0.6,
                 fab_rate: float = 0.10, dis_rate: float = 0.25, vis_min: float = 0.6, omit_rate: float = 0.10,
                 attn_chars: int | None = None, note_keep: float = 0.25):
        super().__init__(read or ReadModel(budget_chars=80_000, base_err=0.01, load_err=0.04, chatty=0.2))
        self.reliable_frac, self.miss_rate, self.omit_rate = reliable_frac, miss_rate, omit_rate
        self.attn_chars = int(attn_chars or self.read.budget_chars)
        self.note_keep = note_keep

        self.fab_rate, self.dis_rate, self.vis_min = fab_rate, dis_rate, vis_min

    def _fact_grams(self, hidden: dict) -> list:
        key = hash(tuple(f["text"] for f in hidden["facts"]))
        g = _FACT_CACHE.get(key)
        if g is None:
            g = [_grams(f["text"]) for f in hidden["facts"]]
            _FACT_CACHE[key] = g
            while len(_FACT_CACHE) > 256:
                _FACT_CACHE.popitem(last=False)
        return g

    def visible(self, prompt: str, hidden: dict) -> list[dict]:
        pg = set(_grams(prompt))
        out = []
        for f, gs in zip(hidden["facts"], self._fact_grams(hidden), strict=True):
            if gs and sum(1 for g in gs if g in pg) >= self.vis_min * len(gs):
                out.append(f)
        return out

    def instruction(self, prompt: str, hidden: dict) -> str:
        """The root's own words: the prompt with all context text removed."""
        res = hidden["_index"].residual(prompt)
        ctx = self._ctx_pieces(hidden)
        return "\n".join(ln for ln in res.split("\n") if ln.strip() and _WS.sub(" ", ln).strip() not in ctx)

    def _ctx_pieces(self, hidden: dict) -> set:
        key = ("pieces", hash(tuple(f["text"] for f in hidden["facts"])), len(hidden["items"]))
        got = _FACT_CACHE.get(key)
        if got is None:
            got = set()
            for text in [t for _, t in hidden["items"]] + list(hidden.get("noise_lines") or ()):
                for piece in _SPLIT.split(text):
                    w = _WS.sub(" ", piece).strip()
                    if w:
                        got.add(w)
            _FACT_CACHE[key] = got
            while len(_FACT_CACHE) > 256:
                _FACT_CACHE.popitem(last=False)
        return got

    def load(self, prompt: str) -> float:
        return min(1.0, len(prompt) / self.attn_chars)

    def requested(self, instr: str, fields: list[dict]) -> list[dict]:
        """Fields named in the root's instruction (aliases only if no exact names appear)."""
        one = re.search(r'\bField:\s*"([^"\n]+)"', instr) or re.search(r'\bfield\s+"([^"\n]+)"', instr, re.I)
        if one:
            hit = [f for f in fields if norm(f["name"]) == norm(one.group(1))]
            if hit:
                return hit
        low = " " + norm(instr) + " "
        exact = [f for f in fields if f" {norm(f['name'])} " in low]
        if exact:
            return exact
        return [f for f in fields if any(f" {norm(a)} " in low for a in ALL_FIELDS[f["key"]][3])]

    def respond(self, prompt: str, hidden: dict, st: dict) -> str:
        fields = hidden["fields"]
        for ov in st.get("overviews") or ():
            prompt = prompt.replace(ov, " ")
        ids = self.given(prompt, hidden)
        reads = st.setdefault("read_ids", {})
        vis = self.visible(prompt, hidden)
        instr = self.instruction(prompt, hidden)
        if self._is_probe(instr):
            return self._overview(prompt, hidden, st, vis)
        req = self.requested(instr, fields)
        L = self.load(prompt)
        if not ids and not vis and not re.search(r"notes on it", instr, re.I):
            return refuse_aggregate(st)
        p_skip = self.miss_rate * max(0.0, L - self.reliable_frac) / max(1e-9, 1 - self.reliable_frac)
        kept = []
        ev = st.setdefault("evidence_read", {})
        for f in vis:
            if p_skip > 0 and rng_for(prompt, f["id"], "skip").random() < p_skip:
                st["items_skipped"] += 1
                continue
            kept.append(f)
            if f["kind"] == "true":
                ev[str(f["id"])] = 1
        for iid in ids:
            reads[iid] = reads.get(iid, 0) + 1
        st["items_read"] = len(reads)
        if not req:
            return self._summary(prompt, kept)
        want_q = bool(re.search(r"\bquot(?:e|es|ed|ation|ations)\b|verbatim|supporting (?:text|evidence|sentence|passage)|"
                                r"exact (?:text|words|sentence|wording)|cite the (?:text|sentence|passage)", instr, re.I))
        want_d = bool(re.search(r'"date"\s*:', instr))
        dated = bool(re.search(r"include (?:the )?dates|with (?:its|their|the) dates?|dates? \(mm/dd/yyyy\)", instr, re.I))
        n_req = len(req)
        pairs = []
        style = rng_for(prompt, "style").randrange(4)
        segs = field_segs(instr.lower(), req)
        reduce = n_req == 1 and bool(re.search(r"findings|candidates|notes on it", instr, re.I))
        c6_reduce = reduce and not want_q and bool(re.search(r"notes on it", instr, re.I))
        notes = re.findall(r"(?m)^- (.+)$", re.split(r"notes on it", prompt, maxsplit=1, flags=re.I)[-1].split(
            "\n\n", 1)[0]) if c6_reduce else []
        plow = prompt.lower()
        listing = None
        for mk in ("organisations of the plaintiffs' attorneys", "plaintiff party/counsel listing"):
            if mk in plow:
                listing = plow.split(mk, 1)[1]
        skip_ok = bool(re.search(r"skip (?:the )?fields|omit (?:the )?fields|says nothing about|fields? (?:it|this part) "
                                 r"gives information on", instr, re.I))
        self._ctx = {"skip_ok": skip_ok, "listing": listing, "instr_low": instr.lower(), "instr_dates": set(find_dates(instr)),
                     "rul_dates": {x["val"][0] for x in kept if x["key"] == "rulings"}}
        for f in req:
            fk = [x for x in kept if x["key"] == f["key"]]
            if reduce and f["type"] in VALUE_READ_TYPES:
                fk += self._value_echo(instr, f, hidden, {x["id"] for x in fk})
            elif c6_reduce:
                fk += self._note_echo(prompt, f, hidden, {x["id"] for x in fk})
            val, qf = self._field(prompt, f, fk, n_req, L, st, segs.get(f["name"], ""), reduce)
            pairs.append((f, val, qf))
        dctx = (dated, hidden.get("variant"))
        bare = n_req == 1 and bool(re.search(r"only the value|no field name|just the value|value only", instr, re.I))
        if bare:
            f, val, sup = pairs[0]
            if val is None and notes and rng_for(prompt, f["key"], "notekeep").random() < self.note_keep:
                return notes[0].strip()
            return self._val(f, val, sup, style, dctx)
        return self._render(prompt, instr, pairs, want_q, want_d, style, dctx)

    def _field(self, prompt: str, f: dict, facts: list[dict], n_req: int, L: float, st: dict | None = None,
               seg: str = "", reduce: bool = False):
        """(value or None, supporting facts) for one field over the visible evidence; confabulations are logged in ``st``."""
        st = st if st is not None else {}
        key, rule = f["key"], f["rule"]
        r = rng_for(prompt, key, "field")
        load_k = (0.4 + L)
        p_omit = self.omit_rate * min(1.5, n_req / 20) * (0.2 + L)
        facts = [x for x in facts if rng_for(prompt, key, x["id"], "omit").random() >= p_omit]
        good = [x for x in facts if x["kind"] in ("true", "stale")]
        dist = [x for x in facts if x["kind"] == "distractor" and not x.get("trap")]
        traps = [x for x in facts if x["kind"] == "distractor" and x.get("trap")]
        mode = def_mode(key, seg) if (traps or key in ("judge", "related")) else "none"
        rt = rng_for(prompt, key, "trap")
        ctx = getattr(self, "_ctx", None) or {}

        def tmode(tp: dict) -> str:
            return def_mode(TRAP_DEF.get(tp.get("trap"), key), seg)
        if key == "settlement_date" and "consent decree" not in seg:
            good = [x for x in good if not (re.search(r"consent decree", x["text"], re.I)
                                            and rng_for(prompt, x["id"], "decr").random() < 0.7)]
        if key == "settlement_date" and not reduce and n_req >= 4:
            good = [x for x in good if not (re.search(r"final approval|approving the parties", x["text"], re.I)
                                            and rng_for(prompt, x["id"], "appr").random() < P_MISS_APPROVAL)]
        over = [x for x in traps if x["trap"] in OVERRIDE_RATE]
        if over and good and rule != "union":
            tp = over[0]
            rates = OVERRIDE_RATE[tp["trap"]]
            if rt.random() < rates.get(tmode(tp), rates["none"]):
                st.setdefault("distractor_picks", []).append([key, tp["val"]])
                return self._describe(f, tp["val"], [tp], seg), [tp]
        if key == "judge" and mode == "no_mag" and rt.random() < 0.9:
            good = [x for x in good if "magistrate" not in x["text"].lower()]
        if traps and key == "filing_date":
            tp = traps[0]
            if (not good and rt.random() < 0.7) or (good and rt.random() < TRAP_RATE["transfer"][mode]):
                st.setdefault("distractor_picks", []).append([key, tp["val"]])
                return tp["val"], [tp]
        elif traps and not good:
            tp = traps[0]
            rates = TRAP_RATE.get(tp["trap"]) or OVERRIDE_RATE.get(tp["trap"], {})
            if rt.random() < rates.get(tmode(tp), rates.get("none", 0.0)):
                v = tp["val"] if rule != "union" else [tp["val"]]
                st.setdefault("distractor_picks", []).append([key, v])
                return v, [tp]
        if not good:
            if dist and r.random() < self.dis_rate * load_k:
                d = r.choice(dist)
                st.setdefault("distractor_picks", []).append([key, d["val"] if rule != "union" else [d["val"]]])
                return (d["val"] if rule != "union" else [d["val"]]), [d]
            if r.random() < self.fab_rate * min(1.0, n_req / 12) * load_k * (FAB_SKIP_OK if ctx.get("skip_ok") else 1.0):
                fab = self._fabricate(prompt, f)
                st.setdefault("fabricated", []).append([key, fab])
                return fab, [{"text": self._fake_quote(prompt, f), "date": ""}]
            return None, []
        if rule == "union":
            vals, sup = [], []
            for x in good:
                if x["kind"] != "true":
                    continue
                if x["val"] not in vals:
                    vals.append(x["val"])
                    sup.append(x)
            if dist and r.random() < 0.5 * self.dis_rate * load_k:
                d = r.choice(dist)
                if d["val"] not in vals:
                    vals.append(d["val"])
                    sup.append(d)
                    st.setdefault("distractor_picks", []).append([key, [d["val"]]])
            n_trap: dict = {}
            for x in traps:
                tab = REDUCE_KEEP if reduce else TRAP_RATE
                rates = tab.get(x["trap"], TRAP_RATE.get(x["trap"], {}))
                if x["trap"] in ("routine", "names") and n_trap.get(x["trap"], 0) >= ROUTINE_CAP:
                    continue
                if x["val"] not in vals and rng_for(prompt, key, x["id"], "trap").random() < rates.get(tmode(x), rates.get("none", 0.0)):
                    vals.append(x["val"])
                    sup.append(x)
                    n_trap[x["trap"]] = n_trap.get(x["trap"], 0) + 1
                    st.setdefault("distractor_picks", []).append([key, [x["val"]]])
            if key == "counsel_type" and "Pro se" in vals and len(vals) > 1 \
                    and def_mode("counsel_prose", seg) != "excl" and rt.random() < 0.5:
                i = vals.index("Pro se")
                vals.pop(i)
                sup.pop(i)
            if reduce and key in ("counsel_type", "counsel") and ctx.get("listing") is not None:
                lst = ctx["listing"]
                keep = [(v, x) for v, x in zip(vals, sup, strict=True) if not isinstance(v, str)
                        or (("pro se" if v == "Pro se" else v.lower()) in lst)
                        or rng_for(prompt, key, v, "listing").random() >= P_TRUST_LISTING]
                vals, sup = [v for v, _ in keep], [x for _, x in keep]
            if f["type"] == "events" and key == "filings":
                out_ok = def_mode("filings_outcome", seg) == "outcome"
                vals = [v if (len(v) < 6 or (out_ok and (v[3] in ctx.get("rul_dates", ()) or (
                    reduce and v[3] in ctx.get("instr_dates", ()) and v[5] in ctx.get("instr_low", ""))))) else v[:3]
                        for v in vals]
            if not vals:
                return None, []
            return vals, sup
        if rule == "earliest":
            pick = min(good, key=lambda x: (x["date"], x["kind"] != "true"))
        elif rule == "latest":
            if key == "appeal":
                merged, sup = {}, []
                for x in sorted(good, key=lambda x: x["date"]):
                    merged.update(x["val"])
                    sup.append(x)
                return merged, sup
            pick = max(good, key=lambda x: (x["date"], x["kind"] == "true"))
        else:
            trues = [x for x in good if x["kind"] == "true"] or good
            pick = trues[0]
        val = pick["val"]
        if r.random() < self.err_prob(prompt):
            if dist:
                d = r.choice(dist)
                st.setdefault("distractor_picks", []).append([key, d["val"]])
                return d["val"], [d]
            val = self._perturb(prompt, f, val)
        if key == "court_enforced" and val == "Yes" and not re.search(r"consent decree|settlement", pick["text"], re.I) \
                and def_mode("court_enforced", seg) != "inj" and rt.random() < P_NARROW_ENFORCED:
            st.setdefault("distractor_picks", []).append([key, "No"])
            return "No", [pick]
        return self._describe(f, val, [pick], seg), [pick]

    @staticmethod
    def _describe(f: dict, val, sup: list, seg: str):
        """'Individual, described as <who>' when the definition asks for a description."""
        if f["key"] != "class_action" or val != "Individual" or def_mode("class_desc", seg) != "describe":
            return val
        for x in sup:
            m = re.search(r"Plaintiffs?, (.+?) at ", x.get("text", ""))
            if m:
                return f"Individual, described as {m.group(1)}"
        return val

    def _value_echo(self, instr: str, f: dict, hidden: dict, have: set) -> list[dict]:
        """Facts whose values are restated in the instruction (a reduce reader reads values, not just quotes)."""
        low = instr.lower()
        dates = None
        out = []
        for x in hidden["facts"]:
            if x["key"] != f["key"] or x["id"] in have:
                continue
            v = x["val"]
            if f["type"] == "events":
                if dates is None:
                    dates = set(find_dates(instr))
                ok = v[0] in dates and v[2].lower() in low
            elif isinstance(v, str) and ":" in v and f["type"] == "parties":
                ok = v.split(":", 1)[1].lower() in low
            elif isinstance(v, str) and ":" in v and f["type"] in ("cause", "terms"):
                kind, body = v.split(":", 1)
                ok = (STATUTES[body][0][0].lower() in low) if kind == "claim" and body in STATUTES else body.lower() in low
            elif isinstance(v, str):
                ok = v.lower() in low
            else:
                ok = False
            if ok:
                out.append(x)
        return out

    def _note_echo(self, instr: str, f: dict, hidden: dict, have: set) -> list[dict]:
        """Facts of this field whose value (in any of the reader's own render styles) a values-only note restates."""
        src = re.split(r"notes on it", instr, maxsplit=1, flags=re.I)[-1]
        low = " " + norm(src) + " "
        short = 2 if f["type"] in ("yesno", "enum", "int") else 4
        out = []
        for x in hidden["facts"]:
            if x["key"] != f["key"] or x["id"] in have or x["kind"] not in ("true", "stale", "distractor"):
                continue
            val = [x["val"]] if f["rule"] == "union" and f["type"] not in ("appeal",) else x["val"]
            try:
                forms_ = {norm(self._fmt(f, val, st_)) for st_ in range(4)}
            except Exception:  # noqa: BLE001
                continue
            if any(v and len(v) >= short and f" {v} " in low for v in forms_):
                out.append(x)
        return out

    @staticmethod
    def _is_probe(instr: str) -> bool:
        """Is this the overview probe (a request over file samples with no field list)?"""
        low = instr.lower()
        return bool(re.search(r"\bin at most \d+ words\b|\boverview\b|\bstructure\b", low)
                    and re.search(r"caption|docket sheet|kinds of documents|which documents", low)
                    and "fields:" not in low)

    def _overview(self, prompt: str, hidden: dict, st: dict, vis: list[dict]) -> str:
        """Deterministic <= 120-word case overview from the given samples."""
        r = rng_for(prompt, "overview")
        cap = re.search(r"(?m)^(.{3,160}? v\. .{3,160}?)(?: Date Filed:|$)", prompt)
        court = re.search(r"District of [A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?", prompt)
        dno = re.search(r"DOCKET FOR CASE #:\s*(\S+)", prompt) or re.search(r"\b\d:\d\d-[a-z]{2}-\d+\b", prompt)
        dates = sorted(find_dates(prompt))
        sheet = "DOCKET FOR CASE" in prompt
        kinds = [k for k, rx in (("complaint", r"complaint"), ("court opinions", r"opinion|memorandum and order"),
                                 ("orders", r"\border\b"), ("a consent decree", r"consent decree"),
                                 ("a settlement agreement", r"settlement"), ("appeal papers", r"appeal"),
                                 ("monitor reports", r"monitor'?s? (?:report|status)"), ("motions", r"\bmotion"))
                 if re.search(rx, prompt, re.I)]
        out = []
        cap_s = cap.group(1).strip()[:120] if cap else "a civil-rights case"
        out.append(f"This is {cap_s}" + (f", in the U.S. District Court for the {court.group(0)}" if court else "")
                   + (f", docket no. {dno.group(1) if dno.re.groups else dno.group(0)}" if dno else "")
                   + (f", filed in {dates[0][:4]}" if dates else "") + ".")
        out.append("The file contains a PACER docket sheet with the docket entries." if sheet else
                   "No PACER docket sheet appears in the samples; the file is a set of case documents.")
        if kinds:
            out.append("Documents seen: " + ", ".join(kinds) + ".")
        if dates:
            out.append(f"Dates in the samples run from {dates[0][:4]} to {dates[-1][:4]}.")
        oc = [x for x in vis if x["kind"] == "true" and x["key"] in ("settlement_date", "court_enforced", "decree_terms",
                                                                      "decree_dates", "rulings")]
        if oc and r.random() < 0.8:
            t = oc[-1]["text"]
            out.append("Outcome visible: " + (t[:200] + ("..." if len(t) > 200 else "")))
        else:
            out.append("The outcome is not visible in these samples.")
        ov = " ".join(out)
        st.setdefault("overviews", []).append(ov)
        return ov

    def _fabricate(self, prompt: str, f: dict):
        r = rng_for(prompt, f["key"], "fab")
        t = f["type"]
        from .textkit import FIRST, LAST
        if t in ("date",):
            return f"{r.randint(1997, 2021)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"
        if t in ("dates",):
            return [f"{r.randint(1997, 2021)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"]
        if t in ("person",):
            return [r.choice(FIRST), None, r.choice(LAST)]
        if t == "persons":
            return [[r.choice(FIRST), None, r.choice(LAST)]]
        if t == "amount":
            return r.randint(4, 200) * 25_000
        if t == "duration":
            return r.choice([12, 24, 36, 60])
        if t == "yesno":
            return r.choice(["Yes", "No"])
        if t == "enum":
            return r.choice(["Class Action", "Individual"]) if f["key"] == "class_action" else \
                r.choice(["former", "never", "current", "female", "male"])
        if t == "casenos":
            return [[r.randint(1, 4), r.randint(0, 21), r.randint(100, 9999)]]
        if t == "cites":
            return [["wl", r.randint(1998, 2021), r.randint(100000, 9999999)]]
        if t == "appeal":
            return {"circuit": r.randint(1, 11), "outcome": r.choice(["affirmed", "reversed"])}
        if t == "statutes":
            return [r.choice(list(STATUTES))]
        if t == "remedies":
            return [r.choice(list(REMEDIES))]
        if t in ("int",):
            return r.randint(1, 9)
        if t == "days":
            return r.choice([3, 5, 10, 14])
        if t == "bp":
            return f"{r.randint(110, 150)}/{r.randint(70, 95)}"
        if t in ("orgs", "entities", "words", "counsel_type"):
            return [f"{r.choice(LAST)} {r.choice(['Legal Group', 'Associates', 'Partners', 'Center'])}"]
        if t == "cause":
            return [f"conduct:{r.choice(CONDUCT)}"]
        if t == "events":
            return [[f"{r.randint(1997, 2021)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}", "grant",
                     "granted the motion"]]
        if t == "terms":
            return [f"remedial:{r.choice(TOPICS)}"]
        if t == "parties":
            return [f"def:{r.choice(LAST)} County"]
        return f"{r.choice(LAST)} {r.choice(['Center', 'Program', 'Facility'])}"

    def _fake_quote(self, prompt: str, f: dict) -> str:
        r = rng_for(prompt, f["key"], "fq")
        return r.choice([
            f"the record reflects that the {f['name'].lower()} was addressed in the parties' filings",
            f"as noted in the pleadings, the {f['name'].lower()} is set out in the court's order",
            f"the materials indicate the relevant {f['name'].lower()} for this matter",
        ])

    def _perturb(self, prompt: str, f: dict, val):
        r = rng_for(prompt, f["key"], "perturb")
        t = f["type"]
        if t == "date":
            y, m, d = (int(x) for x in val.split("-"))
            return f"{y}-{m:02d}-{min(28, max(1, d + r.choice([-9, -3, 3, 11]))):02d}"
        if t == "amount":
            return int(val * r.choice([0.8, 1.1, 1.25]))
        if t == "duration" and not isinstance(val, str):
            return val + 12
        if t == "person":
            from .textkit import LAST
            return [val[0], val[1], r.choice(LAST)]
        if t == "int":
            return max(0, val + r.choice([-1, 1]))
        return val

    def _fmt(self, f: dict, val, style: int) -> str:
        t = f["type"]
        if val is None:
            return "N/A"
        if t == "date":
            from .textkit import from_iso
            return fmt_date(from_iso(val), style)
        if t == "dates":
            from .textkit import from_iso
            return "; ".join(fmt_date(from_iso(v), style) for v in sorted(val))
        if t == "person":
            return fmt_person(val, style % 2, "Dr." if (f["key"] == "monitor" and style == 3) else "")
        if t == "persons":
            return "; ".join(fmt_person(p, style % 2) for p in val)
        if t == "amount":
            return fmt_amount(int(val), style)
        if t == "duration":
            return val if isinstance(val, str) else fmt_duration(int(val), style)
        if t == "casenos":
            return "; ".join(f"{c[1]:02d}-{c[2]}" if c[0] == "usca" else fmt_caseno(c, style % 2) for c in val)
        if t == "counsel_type":
            orgs = [str(x) for x in val if x != "Pro se"]
            if "Pro se" in val:
                return "Pro se" + (", later represented by " + "; ".join(orgs) if orgs else "")
            return "; ".join(orgs)
        if t == "parties":
            grp = [x[6:] for x in val if x.startswith("group:")]
            nm = [x[5:] for x in val if x.startswith("name:")]
            dfs = [x[4:] for x in val if x.startswith("def:")]
            pl = ", ".join(nm) + (f" ({grp[0]})" if grp and nm else "") if nm else (grp[0] if grp else "")
            return "; ".join(x for x in ([f"Plaintiff: {pl}"] if pl else []) + ([f"Defendants: {', '.join(dfs)}"] if dfs else []))
        if t == "cause":
            conduct = [x.split(":", 1)[1] for x in val if x.startswith("conduct:")]
            claims = [STATUTES[x.split(":", 1)[1]][0][0] for x in val if x.startswith("claim:")
                      and x.split(":", 1)[1] in STATUTES]
            out = "Suit under " + ", ".join(claims) if claims else "Civil rights suit"
            return out + (f" alleging {conduct[0]}" if conduct else "")
        if t == "events":
            from .textkit import from_iso
            evs = sorted(val, key=lambda e: e[0])
            pre = "the court " if f["key"] == "rulings" else ""
            return "; ".join(f"On {fmt_date(from_iso(e[0]), style)}, "
                             f"{'' if e[2].startswith('the ') or not pre else pre}{e[2]}"
                             + (f", which the court {e[5]} on {fmt_date(from_iso(e[3]), style)}" if len(e) >= 6 else "")
                             for e in evs)
        if t == "terms":
            ph = {"monitor": "a Monitor oversees compliance", "deadline": "remedial measures must be completed by the "
                  "compliance deadline", "deny": "the court denied the motion for a preliminary injunction"}
            return "; ".join(ph.get(x) or (f"Defendants must end the {x[8:]}" if x.startswith("conduct:") else
                                           f"remedial order concerning {x.split(':', 1)[-1]}") for x in val)
        if t == "duration" and isinstance(val, str):
            return val
        if t == "cites":
            return "; ".join(fmt_cite(c) for c in val)
        if t == "statutes":
            return "; ".join(STATUTES[k][0][0] for k in val)
        if t == "remedies":
            return "; ".join(REMEDIES[k][0] for k in val)
        if t == "appeal":
            parts = []
            if "circuit" in val:
                parts.append(f"appealed to the U.S. Court of Appeals for the {ORDINAL[val['circuit']]} Circuit")
            if "outcome" in val:
                parts.append(f"the appellate court {val['outcome']}")
            return "; ".join(parts) or "N/A"
        if t == "days":
            v = int(val)
            return f"{v} days" if (f["key"] == "onset" or v % 7) else f"{v // 7} weeks"
        if isinstance(val, list):
            return "; ".join(str(x) for x in val)
        return str(val)

    def _val(self, f: dict, val, sup: list, style: int, dctx: tuple) -> str:
        """Value text; with "include dates" in the prompt, a latest-wins docket value carries its evidence date."""
        v = self._fmt(f, val, style)
        dated, variant = dctx
        if (dated and val is not None and sup and f["rule"] == "latest" and f["type"] not in ("date", "dates")
                and variant == "docket" and sup[-1].get("date")):
            from .textkit import from_iso
            v += f" ({fmt_date(from_iso(sup[-1]['date']), 0)})"
        return v

    def _render(self, prompt: str, instr: str, pairs: list, want_q: bool, want_d: bool, style: int,
                dctx: tuple) -> str:
        r = rng_for(prompt, "render")
        as_json = wants_json(instr)
        if as_json:
            obj = {}
            for f, val, sup in pairs:
                v = self._val(f, val, sup, style, dctx)
                if want_q or want_d:
                    ent = {"value": v}
                    if want_q:
                        nq = 8 if f["type"] in ("events", "terms", "cause", "counsel_type", "casenos") else 3
                        ent["quote"] = " ... ".join(s["text"][:320] for s in sup[:nq]) if val is not None else ""
                    if want_d:
                        ent["date"] = (sup[-1].get("date") or "") if (sup and val is not None) else ""
                    obj[f["name"]] = ent
                else:
                    obj[f["name"]] = v
            body = json.dumps(obj, ensure_ascii=False, indent=1 if r.random() < 0.5 else None)
            if r.random() < self.read.chatty:
                body = "```json\n" + body + "\n```"
            return self.chat(prompt, body)
        lines = []
        for f, val, sup in pairs:
            line = f"{f['name']}: {self._val(f, val, sup, style, dctx)}"
            if want_q and val is not None and sup:
                line += f' (quote: "{sup[0]["text"][:320]}")'
            lines.append(line)
        return self.chat(prompt, "\n".join(lines))

    def _summary(self, prompt: str, facts: list[dict]) -> str:
        r = rng_for(prompt, "summary")
        if not facts:
            return self.chat(prompt, "This excerpt contains routine procedural material (notices, scheduling, "
                                     "exhibits); I don't see anything substantive in it.")
        pick = r.sample(facts, min(len(facts), 4))
        bits = [p["text"][:140] for p in pick]
        return self.chat(prompt, "Summary of this excerpt: " + " / ".join(bits))


def parse_checklist(pred: str) -> dict[str, str]:
    """Same as expertlongbench_real._parse_checklist."""
    txt = (pred or "").strip()
    cands = []
    fence = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
    if fence:
        cands.append(fence.group(1).strip())
    first, last = txt.find("{"), txt.rfind("}")
    if first != -1 and last > first:
        cands.append(txt[first:last + 1])
    cands.append(txt)
    for c in cands:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return {str(k): ("" if v is None else str(v)) for k, v in obj.items()}
    out: dict[str, str] = {}
    for line in txt.splitlines():
        m = re.match(r"\s*[-*]?\s*([^:]{2,80}?)\s*:\s*(.+)$", line)
        if m:
            out[m.group(1).strip().strip('"')] = m.group(2).strip()
    return out


_STAT_PATS = {
    "42usc1983": r"1983", "ada": r"12132|americans with disabilities|\bada\b", "rehab504": r"\b794\b|rehabilitation act|section 504",
    "titlevii": r"2000e|title vii", "fha": r"3604|fair housing", "amend8": r"eighth amendment|8th amendment|cruel and unusual",
    "amend14dp": r"due process", "amend14ep": r"equal protection", "amend1": r"first amendment|1st amendment|free speech",
    "amend4": r"fourth amendment|4th amendment|unreasonable seizure", "idea": r"\b1400\b|individuals with disabilities education|\bidea\b",
    "apa": r"\b706\b|administrative procedure", "28usc1331": r"1331", "28usc1343": r"1343", "28usc2201": r"2201",
    "42usc1988": r"1988",
}
_REM_PATS = {"declaratory": r"declarat", "injunctive": r"injunct", "punitive": r"punitive",
             "damages": r"compensatory|money damages|(?<!punitive )damages", "fees": r"\bfees?\b"}


def _f1(pred: set, gold: set) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def _items(s: str) -> list[str]:
    return [x.strip() for x in re.split(r";|\n|\|", s) if x.strip()]


def field_score(f: dict, cand: str, hidden: dict) -> float:
    t, g, key = f["type"], f["gold"], f["key"]
    c = str(cand)
    if t == "date":
        ds = find_dates(c)
        return 1.0 if ds and ds[0] == g else 0.0
    if t == "dates":
        return _f1(set(find_dates(c)), set(g))
    if t == "person":
        if not person_match(c, g):
            return 0.0
        return 1.0 if len(_items(c)) <= 1 and " and " not in c.lower() else 0.5
    if t == "persons":
        hits = sum(1 for p in g if person_match(c, p))
        dis = [x["val"] for x in hidden["facts"] if x["key"] == key and x["kind"] == "distractor"]
        fp = sum(1 for p in dis if person_match(c, p) and p not in g)
        return _f1(set(range(hits + fp)), set(range(hits)) | set(range(10**6, 10**6 + len(g) - hits))) if (hits + fp) else 0.0
    if t == "amount":
        am = find_amounts(c)
        return 1.0 if am and abs(am[0] - g) <= 0.005 * g else 0.0
    if t == "duration":
        du = find_durations(c)
        return 1.0 if du and abs(du[0] - g) < 0.01 else 0.0
    if t == "yesno":
        w = norm(c).split()
        return 1.0 if w and w[0] == g.lower() else 0.0
    if t == "enum":
        low = c.lower()
        if key == "class_action":
            got = "Class Action" if ("class" in low and "individual" not in low) else \
                ("Individual" if ("individual" in low and "class action" not in low) else None)
            if got != g:
                return 0.0
            return _desc_score(c, f["desc"]) if (g == "Individual" and f.get("desc")) else 1.0
        if key == "sex":
            got = "female" if re.search(r"\bfemale|woman\b", low) else ("male" if re.search(r"\bmale|\bman\b", low) else None)
            return 1.0 if got == g else 0.0
        if key == "smoking":
            got = [k for k, p in {"never": r"never|non-?smoker", "former": r"former|quit|ex-", "current": r"current|smokes|daily|pack"}.items()
                   if re.search(p, low)]
            return 1.0 if got[:1] == [g] else 0.0
        return 1.0 if norm(c) == norm(g) else 0.0
    if t == "casenos":
        return _f1(find_casenos(c) | find_usca(c), {f"usca:{x[1]:02d}-{x[2]}" if x[0] == "usca" else
                                                    f"{x[0]}:{x[1]:02d}-cv-{x[2]}" for x in g})
    if t == "counsel_type":
        return _counsel_type_score(c, g, hidden, key)
    if t == "parties":
        return _parties_score(c, g)
    if t == "cause":
        return _cause_score(c, g)
    if t == "events":
        return _events_score(c, g, hidden, key)
    if t == "terms":
        return _terms_score(c, g)
    if t == "cites":
        return _f1(find_cites(c), {cite_key(x) for x in g})
    if t == "statutes":
        low = c.lower()
        return _f1({k for k, p in _STAT_PATS.items() if re.search(p, low)}, set(g))
    if t == "remedies":
        low = c.lower()
        return _f1({k for k, p in _REM_PATS.items() if re.search(p, low)}, set(g))
    if t in ("orgs", "entities"):
        hits = sum(1 for o in g if token_overlap(c, o) >= 0.75)
        dis = {x["val"] for x in hidden["facts"] if x["key"] == key and x["kind"] == "distractor" and x["val"] not in g}
        fp = sum(1 for o in dis if token_overlap(c, o) >= 0.75)
        n_items = len(_items(c))
        fp = max(fp, n_items - len(g) - 1) if n_items > len(g) + 1 else fp
        if hits == 0:
            return 0.0
        p, r = hits / (hits + fp), hits / len(g)
        return 2 * p * r / (p + r)
    if t == "entity":
        if token_overlap(c, g) < 0.6:
            return 0.0
        others = {str(x["val"]) for x in hidden["facts"] if x["key"] == key and x["kind"] != "true" and x["val"] != g}
        return 0.5 if any(token_overlap(c, o) >= 0.75 and token_overlap(o, g) < 0.75 for o in others) else 1.0
    if t == "appeal":
        low = c.lower()
        s = 0.0
        if g.get("circuit") and (ORDINAL[g["circuit"]].lower() in low or re.search(rf"\b{g['circuit']}(st|nd|rd|th|d)\b", low)):
            s += 0.5
        stems = {"affirmed": "affirm", "reversed": "revers", "vacated": "vacat", "dismissed": "dismiss"}
        outs = [o for o in stems.values() if o in low]
        if g.get("outcome") and outs == [stems.get(g["outcome"], g["outcome"])]:
            s += 0.5
        return s
    if t == "int":
        m = re.search(r"\d+", c)
        return 1.0 if m and int(m.group()) == g else 0.0
    if t == "days":
        m = re.search(r"(\d+)\s*(day|week|month)", c.lower())
        if not m:
            return 0.0
        v = int(m.group(1)) * {"day": 1, "week": 7, "month": 30}[m.group(2)]
        return 1.0 if abs(v - g) <= max(1, 0.1 * g) else 0.0
    if t == "words":
        low = c.lower()
        gold = {w.lower() for w in g}
        dis = {str(x["val"]).lower() for x in hidden["facts"] if x["key"] == key and x["kind"] == "distractor"}
        pred = {w for w in gold | dis if w in low}
        return _f1(pred, gold)
    if t == "bp":
        m = re.search(r"(\d{2,3})\s*(?:/|over)\s*(\d{2,3})", c)
        return 1.0 if m and f"{m.group(1)}/{m.group(2)}" == g else 0.0
    return 1.0 if norm(c) == norm(str(g)) else 0.0


_USCA = re.compile(r"(?<![\d:/\-])(\d{2})\s*[-–]\s*(\d{5})\b(?![-/]\d)")


def find_usca(s: str) -> set[str]:
    return {f"usca:{m.group(1)}-{int(m.group(2))}" for m in _USCA.finditer(s or "")}


STEM_RX = {"grant": r"grant", "deny": r"\bden(?:y|ied|ial|ying)", "approv": r"approv", "enter": r"enter|issu|adopt",
           "extend": r"extend|extension", "judgment": r"judgment", "transfer": r"transfer", "affirm": r"affirm",
           "revers": r"revers", "vacat": r"vacat", "dismiss": r"dismiss", "complaint": r"complaint", "amend": r"amend",
           "injunct": r"injunct", "class": r"class", "enforce": r"enforce|contempt|show cause", "appeal": r"appeal",
           "appearance": r"appearance", "extension": r"extension|extend", "response": r"respon|opposition",
           "status": r"status report"}


def _event_items(c: str) -> list[tuple[set, str]]:
    parts = re.split(r";|\n|(?<=[a-z0-9]{2}[.)])\s+(?=[A-Z])", c)
    return [(set(find_dates(p)), p.lower()) for p in parts if p.strip()]


def _event_hit(ev, items) -> bool:
    rx = STEM_RX.get(ev[1], re.escape(ev[1]))
    return any(ev[0] in ds and re.search(rx, low) for ds, low in items)


def _event_credit(ev, items) -> float:
    """1 per gold event; a filing with a ruling counts half for the filing and half for the outcome."""
    if len(ev) < 4:
        return float(_event_hit(ev, items))
    rx, ro = STEM_RX.get(ev[1], re.escape(ev[1])), STEM_RX.get(ev[3], re.escape(ev[3]))
    base = [(ds, low) for ds, low in items if ev[0] in ds and re.search(rx, low)]
    if not base:
        return 0.0
    return 1.0 if any(ev[2] in ds and re.search(ro, low[low.find(" which") if " which" in low else 0:]) for ds, low in base) \
        else 0.5


def _events_score(c: str, g: list, hidden: dict, key: str) -> float:
    """Recall of gold dated events (date and kind in one item); -0.1 per distractor event."""
    items = _event_items(c)
    hits = sum(_event_credit(ev, items) for ev in g)
    gk = {tuple(ev[:2]) for ev in g}
    fps = {tuple(x["val"][:2]) for x in hidden["facts"] if x["key"] == key and x["kind"] == "distractor"} - gk
    fp = sum(1 for ev in fps if _event_hit(ev, items))
    return max(0.0, hits / len(g) - 0.1 * fp)


def _counsel_type_score(c: str, g: dict, hidden: dict, key: str) -> float:
    """Plaintiffs' counsel: public-interest orgs by name plus private counsel; defense firms are false positives."""
    low = c.lower()
    units = [token_overlap(c, o) >= 0.75 for o in g["public"]]
    if g["private"]:
        units.append("private" in low or any(token_overlap(c, o) >= 0.75 for o in g["private"]))
    if g.get("pro_se"):
        units.append(bool(re.search(r"\bpro se\b", low)))
    hits = sum(units)
    if not units or hits == 0:
        return 0.0
    allp = set(g["public"]) | set(g["private"])
    defense = {x["val"] for x in hidden["facts"] if x["key"] == key and x["kind"] == "distractor"} - allp
    fp = sum(1 for o in defense if token_overlap(c, o) >= 0.75)
    p, r = hits / (hits + fp), hits / len(units)
    return 2 * p * r / (p + r)


_NUMW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def _group_units(c: str, n: int, kind: str) -> list[bool]:
    """Plaintiffs as a group: kind and count."""
    low = c.lower()
    units = [kind.split()[-1][:6] in low]
    if n >= 2:
        units.append(bool(re.search(rf"\b({_NUMW[n] if n < len(_NUMW) else n}|{n})\b", low)))
    return units


def _names_listed(c: str, names: list[str]) -> int:
    low = c.lower()
    return sum(1 for nm in names if nm.lower() in low)


def _desc_score(c: str, d: dict) -> float:
    """'Individual, described as ...' (label .5, description .5); -0.25 for a list of names."""
    u = _group_units(c, d["n"], d["kind"])
    s = 0.5 + 0.5 * sum(u) / len(u)
    return max(0.0, s - (0.25 if _names_listed(c, d["names"]) >= 2 else 0.0))


def _parties_score(c: str, g: dict) -> float:
    """Plaintiffs as a group plus every defendant; -0.1 per extra plaintiff name."""
    units = _group_units(c, g["n"], g["kind"]) + [token_overlap(c, d) >= 0.6 for d in g["defendants"]]
    s = sum(units) / len(units)
    return max(0.0, s - 0.1 * max(0, _names_listed(c, g["names"]) - 1))


def _cause_score(c: str, g: list) -> float:
    """Recall of alleged conduct and each claim's statute; -0.1 per jurisdiction statute named as a claim."""
    low = c.lower()
    hits = 0
    for u in g:
        kind, v = u.split(":", 1)
        if kind == "conduct":
            hits += token_overlap(c, v) >= 0.6
        else:
            hits += bool(re.search(_STAT_PATS[v], low))
    jur = sum(1 for k in ("28usc1331", "28usc1343", "28usc2201") if re.search(_STAT_PATS[k], low))
    return max(0.0, hits / len(g) - 0.1 * jur)


def _terms_score(c: str, g: list) -> float:
    low = c.lower()
    hits = 0
    for u in g:
        if u == "monitor":
            hits += "monitor" in low
        elif u == "deadline":
            hits += bool(re.search(r"deadline|complet|remedial measures", low))
        else:
            hits += token_overlap(c, u.split(":", 1)[1]) >= 0.6
    return hits / len(g)


def gold_answer(hidden: dict) -> dict[str, str]:
    """A reference answer per field that scores 1.0."""
    o = CaseFileOracle()
    out = {}
    for f in hidden["fields"]:
        g = f["gold"]
        if g == "N/A":
            out[f["name"]] = "N/A"
        elif f["type"] == "counsel_type":
            v = "; ".join(g["public"]) + (", and private counsel" if g["private"] else "") \
                if g["public"] else "Private counsel"
            out[f["name"]] = ("Pro se, later represented by " + v) if g.get("pro_se") else v
        elif f["type"] == "events":
            out[f["name"]] = o._fmt(f, [[e[0], e[1], e[1]] + ([e[2], e[3], e[3]] if len(e) > 3 else []) for e in g], 0)
        elif f["type"] == "parties":
            out[f["name"]] = f"Plaintiff: {g['group']}; Defendants: {', '.join(g['defendants'])}"
        elif f["key"] == "class_action" and f.get("desc") and g == "Individual":
            out[f["name"]] = f"Individual, described as {f['desc']['group']}"
        else:
            out[f["name"]] = o._fmt(f, g, 0)
    return out


def score(final: str, gold: Any, hidden: dict, state: Any) -> float:
    pred = parse_checklist(final)
    pred_ci = {k.strip().lower(): v for k, v in pred.items()}
    scores, na_hit, na_tot, val_sum, val_tot = {}, 0, 0, 0.0, 0
    for f in hidden["fields"]:
        cand = pred_ci.get(f["name"].strip().lower(), "")
        if f["gold"] == "N/A":
            s = 1.0 if is_na(cand) or (hidden.get("variant") == "clinic" and _NEG.search(str(cand) or "")) else 0.0
            na_tot += 1
            na_hit += s
        elif is_na(cand):
            s = 0.0
            val_tot += 1
        else:
            try:
                s = float(field_score(f, cand, hidden))
            except Exception:  # noqa: BLE001
                s = 0.0
            val_tot += 1
            val_sum += s
        scores[f["name"]] = s
    if state is not None:
        state["elb_field_scores"] = scores
        state["elb_synth"] = {"parsed_fields": len(pred), "na_credit": na_hit / max(1, na_tot),
                              "value_acc": val_sum / max(1, val_tot), "na_share": na_tot / max(1, len(scores))}
    return sum(scores.values()) / max(1, len(scores))


_NEG = re.compile(r"^\s*(no\b|none\b|never\b|denies\b|not\b|nkda\b|negative\b)", re.I)


def _m(name: str, key: str):
    async def m(state: Any = None, **_kw: Any) -> float:
        return float(((state or {}).get("elb_synth") or {}).get(key, 0.0))
    m.__name__ = name
    return m


EXTRA_METRICS = [_m("elb_na_credit", "na_credit"), _m("elb_value_acc", "value_acc"),
                 _m("elb_parsed_fields", "parsed_fields"), _m("elb_na_share", "na_share")]


__all__ = ["CaseFileOracle", "generate", "parse_checklist", "score"]
