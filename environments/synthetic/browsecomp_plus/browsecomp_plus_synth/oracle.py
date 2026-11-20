"""The fake sub-LM for the synthetic BrowseComp-Plus encyclopedia."""

from __future__ import annotations

import json
import re
from typing import Any

from synth_common.oracle import FakeSubLM, ReadModel, _acct, refuse_aggregate, rng_for
from synth_common.segindex import segments

from . import lexicon as L
from .questions import Clue, _resolve_path, value_holds
from .world import World, get_world

STOP = set("""a an the this that these those it its is was were be been being of in on at to for from by with as and or
but not no nor so than then there here which who whom whose what when where why how all any each every both either
neither one ones some such same other another own person persons individual someone something their they them he she
his her him i me my we our you your do does did done has have had having can could would should may might must will
shall clue clues document documents doc docs text below above following given please reply answer line lines exactly
only just also into over under about after before between more most less very per using use based""".split())

ATTR_RX: list[tuple[str, str]] = [
    ("born", r"\bborn\b|\bbirth|birthday"),
    ("town", r"where\b.*\bborn|birthplace|home ?town|native of|born in which town|which town"),
    ("died", r"\bdied\b|\bdeath\b|passed away|\bdie\b"),
    ("prof", r"profession|occupation|\bjob\b|worked as|work as|career|living as|trade"),
    ("father", r"father|parent"),
    ("mother", r"mother|parent"),
    ("nsib", r"sibling|brother|sister|children|only child"),
    ("sibdeath", r"lost a (?:sibling|brother|sister)|(?:sibling|brother|sister)\b.*\bdied|death of (?:a|his|her|their) (?:sibling|brother|sister)"),
    ("spouse", r"marr(?:y|ied|iage)|spouse|wife|husband|wed\b"),
    ("school", r"school|educated|attended|college|studied"),
    ("prize", r"prize|award|medal|honou?r"),
    ("exhib", r"exhibit|showed|shown at|gallery"),
    ("member", r"member|elected|society"),
    ("nworks", r"how many|number of (?:books|paintings|scores|works)|produced \w+ (?:books|paintings|scores)"),
    ("created", r"\bworks?\b|wrote|written|painted|composed|created|title"),
    ("creator", r"author|who (?:wrote|painted|composed|created|made)|creator|by whom"),
    ("illustrator", r"illustrat|drawings"),
    ("publisher", r"publish"),
    ("year", r"what year|which year|\bwhen\b|\byear\b"),
    ("theme", r"\babout\b|subject|depict|theme|centres? on|centers? on"),
    ("reception", r"receiv|reception|review|critic|success"),
    ("stolen", r"stolen|theft|recovered"),
    ("shown", r"exhibit|shown|on show|months"),
    ("founded", r"found|establish|charter|opened|formed|created in"),
    ("formed", r"\bformed\b|created in"),
    ("closed", r"clos|dissolv|shut|ceased"),
    ("river", r"river|banks"),
    ("region", r"region|district|province"),
    ("pop", r"population|inhabitants|census"),
    ("landmark", r"fort|castle|abbey|tower|landmark"),
    ("located", r"located|based in|where is|which town"),
    ("founder", r"founder|founded by|established by"),
    ("debut", r"turned professional|career began|began working|debut"),
    ("natives", r"who was born in|people born in|born there|natives? of|persons? born in"),
]
_SPECIFIC = [("town", "born"), ("natives", "born"), ("natives", "town"), ("sibdeath", "nsib"), ("sibdeath", "died"),
             ("founder", "founded"), ("illustrator", "created"), ("creator", "created"), ("publisher", "created"),
             ("shown", "exhib"), ("shown", "year"), ("closed", "year"), ("founded", "year"), ("stolen", "year"),
             ("died", "year"), ("formed", "founded"), ("debut", "prof"),
             ("nworks", "created"), ("spouse", "year"), ("prize", "year"), ("member", "year"), ("exhib", "year"),
             ("landmark", "founded"), ("creator", "year"), ("illustrator", "year"), ("town", "located")]

AGG_RX = re.compile(r"which (?:one|document|doc|candidate|entity|person|of (?:these|the))|the (?:final )?answer\b|"
                    r"answer (?:to )?(?:the|this) (?:question|puzzle)|identify the|who is (?:this|the)|"
                    r"what is (?:the|its) (?:name|title)|match(?:es)? all|satisf(?:y|ies) (?:all|every)", re.I)
PER_DOC_RX = re.compile(r"for (?:each|every) (?:document|doc|passage|text|candidate)|each (?:document|doc)|per document|"
                        r"one line per|<doc_?id>|\bdoc_?id\b", re.I)


DESCRIBE_RX = re.compile(r"\bdescribe\b|what (?:it|this) is\b|header (?:lines|strings)|markers", re.I)
_HEADER = re.compile(r"^(?:-{3,}|doc_?id:\s*\S+|title:.*|date:.*|…?\[doc truncated\])$")


def _toks(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP and len(t) > 1]


class Reader:
    """What one prompt gives the oracle: present documents (in order) and the facts their given sentences state."""

    def __init__(self, W: World, prompt: str):
        self.W = W
        low = prompt.lower()
        items = W.index.present(low, 0.6)
        self.docs: list[int] = []
        self.facts: dict[int, list[int]] = {}
        seen: set[int] = set()
        for iid in items:
            d = int(iid.split(":", 1)[0])
            if d not in seen:
                seen.add(d)
                self.docs.append(d)
                self.facts[d] = []
            fids = W.sent_facts.get(iid)
            if fids:
                self.facts[d].extend(fids)
        titles = {W.titles[d].strip().lower() for d in self.docs}
        self._titles = titles
        keep = []
        for line in low.splitlines():
            t = line.strip()
            if not t or _HEADER.match(t) or t in titles:
                continue
            pieces = [pc for pc in segments(t, 8) if hash(pc) not in W.sent_hashes]
            if pieces:
                keep.append(" ".join(pieces))
        self.instr_low = "\n".join(keep)

    def is_doc_line(self, line: str) -> bool:
        low = line.lower().strip()
        if _HEADER.match(low) or low in self._titles:
            return True
        segs = segments(low, 8)
        return bool(segs) and sum(1 for sg in segs if hash(sg) in self.W.sent_hashes) * 2 >= len(segs)


class BCPOracle(FakeSubLM):
    def __init__(self, read: ReadModel | None = None, world_args: tuple[str, str] | None = None):
        super().__init__(read or ReadModel(budget_chars=64_000, reliable_items=8, miss_rate=0.5, base_err=0.03,
                                           load_err=0.10, agg_capacity=4, chatty=0.2, min_cov=0.6))
        self._world_args = world_args

    def respond(self, prompt: str, hidden: dict, st: dict) -> str:
        W = get_world(hidden["world"], hidden["scale"])
        R = Reader(W, prompt)
        instr = R.instr_low
        clues = [Clue(**c) for c in hidden["clues"]]
        cmap = self.match_clues(R, prompt, clues)
        attrs = self.asked_attrs(instr, cmap)
        explicit_per_doc = bool(PER_DOC_RX.search(instr)) or _template(prompt) is not None
        if R.docs and not explicit_per_doc and DESCRIBE_RX.search(instr):
            return self.chat(prompt, "It is one encyclopedia-style web page. Header lines, in order: `---`, "
                                     "`doc_id: <number>`, `---`, `---`, `title: <page title>`, `date: <date>`, `---`; "
                                     "the page text follows.")
        agg = (bool(AGG_RX.search(instr)) or self._mentions_question(instr, hidden)) and not explicit_per_doc
        per_doc = True
        if not R.docs:
            st["agg_requests"] += 1 if agg else 0
            return refuse_aggregate(st)
        if agg:
            st["agg_requests"] += 1
            return self.aggregate(prompt, R, hidden, clues, st)
        ids = [W.doc_ids[d] for d in R.docs]
        st["doc_prompts"] = st.get("doc_prompts", 0) + 1
        st["max_docs"] = max(st.get("max_docs", 0), len(ids))
        if st.get("in_batch"):
            st["batched_doc_prompts"] = st.get("batched_doc_prompts", 0) + 1
        read, _skipped = self.read_items(prompt, ids, st)
        readset = set(read)
        rows = []
        for d in R.docs:
            did = W.doc_ids[d]
            if did not in readset:
                continue
            row = self.doc_row(prompt, W, d, R.facts[d], cmap, clues, attrs)
            if row["yes_fids"]:
                yf = st.setdefault("yes_fids", [])
                yf.extend(f for f in row["yes_fids"] if f not in yf)
            rows.append(row)
        if not rows:
            return self.chat(prompt, "I could not find any readable document in the text you sent.")
        return self.format(prompt, instr, rows, per_doc, W, clues)

    def batched(self, prompts: list[str], state: Any) -> list[str]:
        st = _acct(state)
        st["in_batch"] = True
        try:
            return super().batched(prompts, state)
        finally:
            st["in_batch"] = False

    def match_clues(self, R: Reader, prompt: str, clues: list[Clue]) -> dict[int, int]:
        """Map the root's clue numbers to hidden clue indices, from clue lines quoted in the instruction."""
        sigs = []
        for c in clues:
            sigs.append((set(_toks(c.text)), {t for f in c.forms for t in _toks(f)}))
        out: dict[int, int] = {}
        seq = 0
        for line in prompt.splitlines():
            lt = line.strip()
            if len(lt) < 6 or len(lt) > 400 or R.is_doc_line(lt):
                continue
            m = re.match(r"^[\s\-*•]*(?:clue\s*)?\(?(\d{1,2})[\.\):\]]\s+(.*)$", lt, re.I)
            body = m[2] if m else lt
            toks = set(_toks(body))
            if not toks:
                continue
            best, bscore = -1, 0.0
            for j, (tt, ft) in enumerate(sigs):
                if not tt:
                    continue
                ov = len(toks & tt) + 0.7 * len(toks & ft)
                score = ov / max(1.0, min(len(tt), len(toks)))
                if ov >= min(2, len(tt)) and score > bscore:
                    best, bscore = j, score
            if best >= 0 and bscore >= 0.5:
                if m:
                    num = int(m[1])
                else:
                    seq += 1
                    num = seq
                if num not in out:
                    out[num] = best
        return out

    def asked_attrs(self, instr: str, cmap: dict[int, int]) -> list[str]:
        if cmap and not re.search(r"\banswer:|\bwhat\b|\bwhich\b|\bwho\b|\bwhen\b|\bwhere\b|\bhow\b|\?", instr):
            return []
        q = instr
        m = re.search(r"answer:?\s*(.*?)(?:\n|then judge|reply|$)", instr, re.S)
        if m and len(m[1]) > 8:
            q = m[1]
        out = []
        for a, rx in ATTR_RX:
            if re.search(rx, q):
                out.append(a)
        for hi, lo in _SPECIFIC:
            if hi in out and lo in out:
                out.remove(lo)
        return out[:4]

    def _mentions_question(self, instr: str, hidden: dict) -> bool:
        ask = _toks(hidden.get("ask_text", ""))
        if not ask:
            return False
        it = set(_toks(instr))
        hits = sum(1 for t in ask if t in it)
        return hits >= max(2, int(0.7 * len(ask))) and len(it) > 3 and sum(
            1 for c in hidden["clues"] if len(set(_toks(c["text"])) & it) >= 3) >= 3

    def verdict(self, W: World, facts: list[int], c: Clue) -> tuple[str, int | None]:
        want = _clue_stype(c)
        best = None
        for fid in facts:
            e, a, v = W.facts[fid]
            if W.ents[e]["type"] != want:
                continue
            if c.attr == "gap":
                if a == "closed":
                    fo = [W.facts[f2][2] for f2 in facts if W.facts[f2][0] == e and W.facts[f2][1] == "founded"]
                    if fo:
                        ok = (v - fo[0]) >= c.args[0]
                        if ok:
                            return "YES", fid
                        best = ("NO", fid)
                continue
            if a != c.attr:
                continue
            ok = value_holds(c.attr, c.op, c.args, v)
            if ok:
                return "YES", fid
            if ok is False and best is None:
                best = ("NO", fid)
        return best if best else ("NOT STATED", None)

    def doc_row(self, prompt: str, W: World, d: int, facts: list[int], cmap: dict[int, int], clues: list[Clue],
                attrs: list[str]) -> dict:
        subj = W.doc_subject[d]
        row: dict[str, Any] = {"doc_id": W.doc_ids[d], "entity": W.ents[subj]["name"] if subj >= 0 else "NONE",
                               "yes": [], "no": [], "ns": [], "quote": "", "answer": "", "values": [], "yes_fids": [],
                               "_facts": facts, "_subj": subj, "_d": d}
        err = self.err_prob(prompt)
        for num, j in sorted(cmap.items()):
            v, fid = self.verdict(W, facts, clues[j])
            r = rng_for(prompt, W.doc_ids[d], num, "v").random()
            if r < err:
                v = {"YES": "NOT STATED", "NO": "YES", "NOT STATED": "YES" if r < err / 3 else "NOT STATED"}[v]
                fid = fid if v != "NOT STATED" else None
            {"YES": row["yes"], "NO": row["no"], "NOT STATED": row["ns"]}[v].append(num)
            if fid is not None and v == "YES":
                row["yes_fids"].append(fid)
                if not row["quote"]:
                    row["quote"] = W.fact_sent[fid]
        if attrs:
            vals = self.attr_values(W, facts, attrs, subj, prompt)
            if vals:
                if rng_for(prompt, W.doc_ids[d], "attr").random() < err:
                    vals = vals[1:] or [("NONE", None)]
                top_attr = W.facts[vals[0][1]][1] if vals[0][1] is not None else ""
                if top_attr in _LIST_ATTRS or "natives" in attrs[:1]:
                    same = [v for v, f in vals if f is not None and (W.facts[f][1] == top_attr or "natives" in attrs[:1])]
                    row["answer"] = "; ".join(dict.fromkeys(same))[:400]
                else:
                    row["answer"] = vals[0][0]
                row["values"] = [v for v, _ in vals][:4]
                if vals[0][1] is not None and not row["quote"]:
                    row["quote"] = W.fact_sent[vals[0][1]]
            else:
                row["answer"] = "NONE"
        if not row["quote"] and facts:
            row["quote"] = W.fact_sent[facts[0]]
        if not cmap and not attrs:
            row["values"] = [self.render_fact(W, fid) for fid in _dedup(facts)[:4]]
        return row

    def attr_values(self, W: World, facts: list[int], attrs: list[str], subj: int, prompt: str) -> list[tuple[str, int]]:
        """Values of the asked attributes stated in the given text, most specific first."""
        named = self.named_entities(W, prompt)
        out: dict[tuple, tuple[str, int, tuple]] = {}
        for fid in _dedup(facts):
            e, a, v = W.facts[fid]
            if a == "town" and "natives" in attrs:
                key = ("natives", e)
                if v in named or not named:
                    out.setdefault(key, (W.ents[e]["name"], fid, (attrs.index("natives"), 0)))
                continue
            if a not in attrs:
                continue
            pri = (attrs.index(a), 0 if e in named else (1 if e == subj else 2))
            val = self.render_value(W, a, v)
            key = (a, e, v[0] if a in ("born",) and isinstance(v, tuple) else str(v))
            if key not in out or len(val) > len(out[key][0]):
                out[key] = (val, fid, pri)
        ranked = sorted(out.values(), key=lambda x: x[2])
        return [(v, f) for v, f, _ in ranked]

    def named_entities(self, W: World, prompt: str) -> set[int]:
        low = W.index.residual(prompt.lower())
        found = set()
        words = re.findall(r"[a-z'\-]+", low)
        for n in (2, 3, 4, 5):
            for i in range(len(words) - n + 1):
                k = " ".join(words[i:i + n])
                e = W.name_index.get(k) or W.name_index.get("the " + k)
                if e is not None:
                    found.add(e)
        return found

    def render_value(self, W: World, a: str, v) -> str:
        E = W.ents
        if a == "born":
            y, m, d = v
            return f"{d} {L.MONTHS[m - 1]} {y}" if m else str(y)
        if a in ("town", "located", "school", "publisher", "creator", "illustrator", "founder", "created",
                 "illustrated", "region"):
            return E[v]["name"]
        if a in ("prize", "exhib"):
            return f"{E[v[0]]['name']} ({v[1]})"
        if a == "member":
            return f"{E[v[0]]['name']} ({L.MONTHS[v[2] - 1]} {v[1]})"
        if a == "spouse":
            return f"{E[v[0]]['name']} (married {v[1]})"
        if a == "nworks":
            return f"{v[0]} works between {v[1]} and {v[2]}"
        if a == "shown":
            return f"{E[v[0]]['name']}, {v[1]}, for {v[2]} months"
        if a == "stolen":
            return f"stolen {v[0]}, recovered {v[1]}"
        if a == "pop":
            return f"{v[0]:,} ({v[1]} census)"
        if a == "landmark":
            return f"{v[0]} (built {v[1]})"
        if a == "reception":
            return "poorly received at first" if v == "poor" else "well received"
        return str(v)

    def render_fact(self, W: World, fid: int) -> str:
        e, a, v = W.facts[fid]
        return f"{W.ents[e]['name']} {a}: {self.render_value(W, a, v)}"

    def field_value(self, prompt: str, W: World | None, clues: list[Clue], row: dict, label: str, kind: str,
                    ph: str) -> str:
        """Fill one template field: a clue verdict for yes/no fields, else the attribute's value."""
        if W is None or kind in ("doc_id", "yes", "contra", "ns", "verdict", "quote") or not label:
            return _field(row, kind)
        facts = row.get("_facts") or []
        words = " ".join(re.findall(r"[a-z0-9]+", label.lower().replace("_", " ")))
        yn = bool(re.search(r"\byes\b|\bno\b|true|false", ph, re.I))
        lt = set(_toks(words))
        best, bs = None, 0.0
        for j, c in enumerate(clues):
            tt = set(_toks(c.text)) | {t for f in c.forms for t in _toks(f)}
            ov = len(lt & tt)
            sc = ov / max(1, len(lt))
            if ov >= min(2, len(lt)) and sc > bs:
                best, bs = j, sc
        if best is not None and bs >= 0.5 and re.search(r"\bname\b", ph, re.I) and re.search(
                r"person|name|who|entity|candidate", words):
            v, _fid = self.verdict(W, facts, clues[best])
            return row["entity"] if v == "YES" else "NONE"
        if yn and best is not None and bs >= 0.5:
            v, _fid = self.verdict(W, facts, clues[best])
            if rng_for(prompt, row["doc_id"], label, "v").random() < self.err_prob(prompt):
                v = {"YES": "NOT STATED", "NO": "YES", "NOT STATED": "NOT STATED"}[v]
            return {"YES": "YES", "NO": "NO", "NOT STATED": "NOT STATED"}[v]
        attrs = [a for a, rx in ATTR_RX if re.search(rx, words)]
        if re.search(r"count|number|how many", ph, re.I) and re.search(r"book|work|painting|score|novel", words):
            attrs = ["nworks"] + attrs
        for hi, lo in _SPECIFIC + [("father", "prof"), ("mother", "prof"), ("father", "mother"), ("mother", "father")]:
            if hi in attrs and lo in attrs and (hi not in ("father", "mother") or hi in words):
                attrs.remove(lo)
        if attrs and attrs != ["year"] and not (kind == "entity" and not set(attrs) - {"created", "town"}):
            vals = self.attr_values(W, facts, attrs[:2], row.get("_subj", -1), prompt)
            if not vals:
                return "NONE" if not yn else "NOT STATED"
            v = vals[0][0]
            if re.search(r"year", words + " " + ph, re.I):
                m = re.search(r"\b(1[5-9]\d\d|20[0-2]\d)\b", v)
                v = m[1] if m else v
            if attrs[0] == "nworks" and re.search(r"count|number", ph, re.I):
                v = v.split()[0]
            return v
        return _field(row, kind)

    def aggregate(self, prompt: str, R: Reader, hidden: dict, clues: list[Clue], st: dict) -> str:
        W = R.W
        facts = [f for d in R.docs for f in R.facts[d]]
        tgt = hidden["target"]
        complete = all(self._supported(W, facts, tgt, c) for c in clues) and self._ask_supported(W, facts, hidden)
        n = len(R.docs)
        r = rng_for(prompt, "agg").random()
        p_ok = 1.0 if n <= self.read.agg_capacity else self.read.agg_capacity / n
        if complete and r < p_ok * (1 - self.err_prob(prompt)):
            return self.chat(prompt, f"Based on the documents provided, the answer is: {hidden['gold']}")
        cands = {}
        for fid in facts:
            e = W.facts[fid][0]
            if W.ents[e]["type"] == hidden["ttype"] and e != tgt:
                cands[e] = cands.get(e, 0) + 1
        if not cands:
            return self.chat(prompt, "None of the documents provided clearly matches all of the clues; I cannot "
                                     "determine the answer from this text.")
        guess = max(sorted(cands), key=lambda e: (cands[e], rng_for(prompt, e).random()))
        return self.chat(prompt, f"Based on the documents provided, the answer is most likely: {W.ents[guess]['name']}")

    def _supported(self, W: World, facts: list[int], tgt: int, c: Clue) -> bool:
        ents = set(_resolve_path(W, tgt, c.role)) if c.role != "self" else {tgt}
        mine = [f for f in facts if W.facts[f][0] in ents]
        return self.verdict(W, mine, c)[0] == "YES"

    def _ask_supported(self, W: World, facts: list[int], hidden: dict) -> bool:
        g = hidden["gold_entity"]
        return any(W.facts[f][0] == g for f in facts)

    def format(self, prompt: str, instr: str, rows: list[dict], per_doc: bool, W: World | None = None,
               clues: list[Clue] | None = None) -> str:
        tmpl = _template(prompt)
        lines = []
        if tmpl:
            for row in rows:
                parts = []
                for label, kind, ph in tmpl:
                    val = self.field_value(prompt, W, clues or [], row, label, kind, ph)
                    parts.append(f"{label}: {val}" if label else val)
                lines.append(" | ".join(parts))
            return self.chat(prompt, "\n".join(lines))
        if re.search(r"\bjson\b", instr):
            obj = {r["doc_id"]: {"entity": r["entity"], "clues": r["yes"], "contradicted": r["no"],
                                 "answer": r["answer"] or None, "quote": r["quote"]} for r in rows}
            return self.chat(prompt, json.dumps(obj, ensure_ascii=False))
        for r in rows:
            seg = [r["doc_id"], f"entity: {r['entity']}"]
            if r["yes"] or r["no"] or r["ns"]:
                seg.append("clues: " + (", ".join(map(str, r["yes"])) or "0"))
                if r["no"]:
                    seg.append("contradicted: " + ", ".join(map(str, r["no"])))
            if r["answer"]:
                seg.append(f"answer: {r['answer']}")
            elif r["values"]:
                seg.append("facts: " + "; ".join(r["values"]))
            if r["quote"]:
                seg.append(f'quote: "{r["quote"]}"')
            lines.append(" | ".join(seg))
        return self.chat(prompt, "\n".join(lines))


_LIST_ATTRS = {"created", "illustrated", "exhib", "prize", "member", "natives"}


def _dedup(xs: list[int]) -> list[int]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


_ROLE_STYPE = {"self": None, "birthplace": "town", "school": "org", "spouse": "person", "works": "work",
               "creator": "person", "illustrator": "person", "publisher": "org", "shown_at": "org", "region": "region",
               "natives": "person", "founder": "person", "located": "town"}
_ATTR_STYPE = {"prof": "person", "born": "person", "father": "person", "mother": "person", "nsib": "person",
               "sibdeath": "person", "died": "person", "spouse": "person", "school": "person", "prize": None,
               "exhib": "person", "member": "person", "nworks": "person", "debut": "person", "theme": "work",
               "year": "work", "reception": "work", "shown": "work", "stolen": "work", "publisher": "work",
               "founded": None, "closed": "org", "gap": "org", "river": "town", "pop": "town", "landmark": "town",
               "formed": "region"}


def _clue_stype(c: Clue) -> str:
    last = c.role.split(".")[-1]
    t = _ROLE_STYPE.get(last)
    if t:
        return t
    st = _ATTR_STYPE.get(c.attr)
    if st:
        return st
    if c.attr == "prize":
        return "person"
    return "town" if c.attr == "founded" else "person"


_FIELD_KINDS = [
    ("doc_id", r"^doc_?id$|^id$|^doc$|^document$"),
    ("contra", r"contradict|fail|violat|reject|mismatch"),
    ("ns", r"missing|unknown|not.?stated|open|unclear"),
    ("verdict", r"satisf|verdict|match(?:es)?$|holds|consistent"),
    ("yes", r"clue|constraint|condition|matched|criteria"),
    ("entity", r"entity|name|subject|candidate|person|who|target|about"),
    ("quote", r"quote|evidence|excerpt|sentence|support|span"),
    ("year", r"year|date"),
    ("answer", r"answer|fact|value|info|detail|summary|note|result|finding"),
]


def _template(prompt: str) -> list[tuple[str, str, str]] | None:
    """Fields of a one-line-per-document template like ``<doc_id> | clues: <...> | entity: <...>``."""
    for line in prompt.splitlines():
        if line.count("<") < 2 or "|" not in line:
            continue
        seg = line
        m = re.search(r"`([^`]*\|[^`]*)`", line)
        if m:
            seg = m[1]
        parts = [p.strip() for p in seg.split("|") if p.strip()]
        out = []
        for p in parts:
            lm = re.match(r"^([A-Za-z_][A-Za-z _\-]{0,30}?)\s*:\s*(.*)$", p)
            if lm:
                label, ph = lm[1].strip(), lm[2]
            else:
                label, ph = "", p
            key = (label or ph).strip("<>[]` ").lower()
            kind = "answer"
            for k, rx in _FIELD_KINDS:
                if re.search(rx, key):
                    kind = k
                    break
            if kind == "yes" and re.search(r"yes|no\b", ph, re.I) and not re.search(r"number", ph, re.I):
                kind = "verdict"
            out.append((label, kind, ph))
        if len(out) >= 2:
            return out
    return None


def _field(row: dict, kind: str) -> str:
    if kind == "doc_id":
        return row["doc_id"]
    if kind == "yes":
        return ", ".join(map(str, row["yes"])) or "0"
    if kind == "contra":
        return ", ".join(map(str, row["no"])) or "none"
    if kind == "ns":
        return ", ".join(map(str, row["ns"])) or "none"
    if kind == "verdict":
        if row["yes"]:
            return "YES"
        if row["no"]:
            return "NO"
        return "NOT STATED"
    if kind == "entity":
        return row["entity"]
    if kind == "quote":
        return f'"{row["quote"]}"' if row["quote"] else "NONE"
    if kind == "year":
        m = re.search(r"\b(1[5-9]\d\d|20[0-2]\d)\b", row["answer"] or "")
        return m[1] if m else "NONE"
    if row["answer"]:
        return row["answer"]
    if row["values"]:
        return "; ".join(row["values"])
    return row["quote"][:120] if row["quote"] else "NONE"


__all__ = ["BCPOracle", "Reader"]
