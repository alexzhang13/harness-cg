"""Fake sub-LMs for the 9 CTC families: answer only from the items in the prompt, with load-dependent noise."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any

from synth_common.oracle import ReadModel, refuse_aggregate, rng_for, wants_json

from .oracle_base import CTCOracle, Seen, View, bump, label_value, render_per_item, render_selection

_TOK = re.compile(r"[a-z0-9]+")
STOP = frozenset("""a an the of in on at to for from by with and or but is are was were be been being this that these
those it its as which who whom whose what when where why how do does did has have had not no nor so than then there their
they them he she his her we our you your i me my can could would should will shall may might must into about over under
after before between during while also more most such any each other some only own same very just one two first""".split())


def ctoks(s: str) -> list[str]:
    return [t for t in _TOK.findall(s.lower()) if t not in STOP and len(t) > 1]


@lru_cache(maxsize=200_000)
def tokset(s: str) -> frozenset:
    return frozenset(ctoks(s))


def jacc(a: str, b: str) -> float:
    x, y = tokset(a), tokset(b)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def has_any(text: str, *pats: str) -> bool:
    return any(re.search(p, text) for p in pats)


def _labels_list(seen: list[Seen]) -> list[str]:
    return [s.label for s in seen]


def _first_sentence_with(text: str, words: set) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    best, score = sents[0] if sents else text[:200], -1
    for s in sents:
        sc = len(tokset(s) & words)
        if sc > score:
            best, score = s, sc
    return best.strip()[:300]


def _pick_json_keys(view: View) -> list[str]:
    from .oracle_base import _first_json_example
    ex = _first_json_example(view.instr)
    return list(ex.keys()) if isinstance(ex, dict) else []


_SINGLE_ID_KEYS = ("id", "doc", "doc_id", "docid", "document", "document_id", "passage", "passage_id", "best", "choice",
                   "pick", "selected", "label", "claim", "paper", "item", "number")


def fill_keys(keys: list[str], slots: dict) -> dict:
    """Fill the JSON keys the instruction asked for from semantic slots (id, ids, answer, verdict, reason, text)."""
    obj = {}
    for k in keys:
        kl = k.lower()
        if "answer" in kl and "answer" in slots:
            obj[k] = slots["answer"]
        elif any(w in kl for w in ("verdict", "same", "label", "decision", "result", "contradict")) and "verdict" in slots:
            obj[k] = slots["verdict"]
        elif any(w in kl for w in ("reason", "why", "explan", "quote", "evidence_text", "rationale")):
            obj[k] = slots.get("reason", "")
        elif "confiden" in kl:
            obj[k] = 0.9
        elif kl in _SINGLE_ID_KEYS and "id" in slots:
            obj[k] = slots["id"]
        elif any(w in kl for w in ("used", "ids", "docs", "documents", "sources", "evidence", "relevant", "passages",
                                   "support", "citations", "cited")) and "ids" in slots:
            obj[k] = slots["ids"]
        elif any(w in kl for w in ("id", "doc")) and ("id" in slots or "ids" in slots):
            obj[k] = slots.get("id", (slots.get("ids") or [None])[0])
        elif any(w in kl for w in ("title", "article", "topic", "subject", "name")) and "text" in slots:
            obj[k] = slots["text"]
        else:
            obj[k] = next((slots[x] for x in ("answer", "verdict", "id", "ids", "text") if x in slots), None)
    return obj


class RetrievalOracle(CTCOracle):
    """nq / hotpotqa: which of the shown documents help answer the question."""

    p_hard = 0.30
    p_other = 0.004
    bridge_alone = 0.45
    cite_drop = 0.15

    def _q_present(self, view: View, q: str) -> bool:
        qt = tokset(q)
        if not qt:
            return False
        it = tokset(view.instr)
        return len(qt & it) >= max(1, math.ceil(0.5 * len(qt)))

    def _judge(self, view: View, f: dict, s: Seen, st: dict) -> bool:
        iid = int(s.iid)
        err = self.err_prob(view.prompt)
        r = rng_for(view.prompt, s.iid, "rel").random()
        if iid in f["gold_ids"]:
            p = 1.0 - err - (0.0 if s.full else 0.25)
            if f.get("bridge") == iid and not any(int(g) in f["gold_ids"] and int(g) != iid for g in view.by_iid):
                p *= self.bridge_alone
            return r < p
        c = rng_for("confuse", f["question"], s.iid).random()
        if iid in f["hard"]:
            return r < self.p_hard * c * c + err
        q = f["question"]
        ov = len(tokset(q) & tokset(view.prompt[s.pos:s.end])) / max(1, len(tokset(q)))
        return r < self.p_other + 0.1 * (ov ** 2) * c + err / 4

    def answer(self, view: View, hidden: dict, st: dict) -> str:
        f = hidden
        low = view.low
        if not view.items:
            return refuse_aggregate(st)
        read, _skipped = self.read_view(view, st)
        if not self._q_present(view, f["question"]):
            pairs = [(s.label, _describe(view.prompt[s.pos:s.end])) for s in read]
            return render_per_item(pairs, view.instr)
        named = set(re.findall(r"\[(\d{1,7})\]|(?:document|doc)\s*#?\s*(\d{1,7})", low) and
                    [x for t in re.findall(r"\[(\d{1,7})\]|(?:document|doc)\s*#?\s*(\d{1,7})", low) for x in t if x])
        other_q = bool(named) and has_any(low, r"\bother\b", r"\banother\b", r"\bsecond\b")
        rel = [s for s in read if self._judge(view, f, s, st) and not (other_q and s.label in named)]
        if has_any(low, r"\b(rate|score)\b", r"scale of", r"\b0\s*(?:-|to)\s*(?:3|5|10)\b",
                   r"\b1\s*(?:-|to)\s*(?:5|10)\b"):
            out = []
            for s in read:
                r = rng_for(view.prompt, s.iid, "score")
                out.append((s.label, r.choice([8, 9, 10] if s in rel else [0, 0, 1, 1, 2, 3])))
            return render_per_item(out, view.instr)
        wants_answer = has_any(low, r"answer the question", r"what is the answer", r"\"answer\"")
        if wants_answer and not has_any(low, r"which (one|single)", r"which (of these )?(documents?|passages?|ids?)\b"):
            gold_here = [s for s in rel if int(s.iid) in f["gold_ids"]]
            used = list(gold_here)
            if len(used) == 2 and f.get("bridge") and rng_for(view.prompt, "drop").random() < self.cite_drop:
                used = [s for s in used if int(s.iid) != f["bridge"]] or used[:1]
            used += [s for s in rel if s not in gold_here and rng_for(view.prompt, s.iid, "cite").random() < 0.3]
            ans = (f.get("answers") or ["(answer)"])[0] if gold_here else "NOT ANSWERABLE"
            if not gold_here and rel and rng_for(view.prompt, "halluc").random() < 0.3:
                ans = _describe(view.prompt[rel[0].pos:rel[0].end])
            labs = [label_value(s.label) for s in used] if ans != "NOT ANSWERABLE" else []
            keys = _pick_json_keys(view)
            if wants_json(view.instr) or keys:
                return json.dumps(fill_keys(keys or ["answer", "used"],
                                            {"answer": ans, "ids": labs, "id": labs[0] if labs else None}))
            return f"{ans} (from {', '.join(map(str, labs)) or 'none of the documents'})"
        single = has_any(low, r"\bwhich (one|single)\b", r"\bmost relevant\b", r"\bbest\b", r"\bone document\b",
                         r"\bsingle document\b", r"\bwhich one other\b", r"\bone other\b")
        if single:
            keys = _pick_json_keys(view)
            if not rel:
                if keys:
                    return json.dumps(fill_keys(keys, {"id": None, "ids": [], "answer": "NOT ANSWERABLE"}))
                return "[]" if wants_json(view.instr) else "None of these documents answers the question."
            n = len(read)
            gold_rel = [s for s in rel if int(s.iid) in f["gold_ids"]]
            p_ok = 0.95 if n <= self.read.agg_capacity else max(0.5, 0.95 - 0.004 * n)
            r = rng_for(view.prompt, "single")
            pick = gold_rel[0] if gold_rel and r.random() < p_ok else r.choice(rel)
            if keys:
                ans = (f.get("answers") or ["(answer)"])[0] if int(pick.iid) in f["gold_ids"] else \
                    _describe(view.prompt[pick.pos:pick.end])
                return json.dumps(fill_keys(keys, {"id": label_value(pick.label), "ids": [label_value(pick.label)],
                                                   "answer": ans}))
            return render_selection([pick.label], view.instr)
        if has_any(low, r"for each|each document|every document|per document|yes or no|yes/no") \
                and not has_any(low, r"list (the|all)|which (documents|ids)"):
            yes, no = _yesno_words(low)
            return render_per_item([(s.label, yes if s in rel else no) for s in read], view.instr)
        return render_selection(_labels_list(rel), view.instr, "NONE")


def _yesno_words(low: str) -> tuple[Any, Any]:
    if "relevant" in low and "irrelevant" in low:
        return "relevant", "irrelevant"
    if re.search(r"\btrue\b", low) and re.search(r"\bfalse\b", low):
        return True, False
    return "YES", "NO"


def _describe(text: str) -> str:
    t = text.strip()
    m = re.match(r'"([^"\n]{2,80})"\s*\n', t)
    if m:
        return m.group(1)
    caps = re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", t)
    if caps:
        return Counter(caps).most_common(1)[0][0]
    return " ".join(t.split()[:6])


class QDMatchOracle(CTCOracle):
    """qdmatch_hpqa: queries and documents in one list; a relevant query needs BOTH its hop documents."""

    p_both = 0.93
    p_cite_both = 0.78
    p_one = 0.45
    p_look = 0.30
    p_look_quote = 0.12
    sens_pairs = 0.92

    def _lookalike(self, f: dict, q: str, qtext: str, docs: list[Seen]) -> list[Seen]:
        low = qtext.lower()
        out = []
        for s in docs:
            ent = (f["ent"].get(s.iid) or "").lower()
            if len(ent) >= 4 and ent in low:
                out.append(s)
        return out

    def answer(self, view: View, hidden: dict, st: dict) -> str:
        f = hidden
        if not view.items:
            return refuse_aggregate(st)
        from .qdmatch_protocol import protocol_answer
        proto = protocol_answer(self, view, hidden, st)
        if proto is not None:
            return proto
        qset = {str(q) for q in f["queries"]}
        read, _ = self.read_view(view, st)
        qs = [s for s in read if s.iid in qset]
        ds = [s for s in read if s.iid not in qset]
        low = view.low
        from .oracle_base import texts_of
        alltexts = texts_of(hidden)
        texts = {s.iid: alltexts.get(s.iid) or view.prompt[s.pos:s.end] for s in read}
        quote_mode = has_any(low, r"quote", r"verbatim", r"chain")
        if quote_mode:
            bump(st, "chain_calls")
        if not qs:
            if ds and has_any(low, r"which (quer|question)"):
                return render_selection([], view.instr, "NONE")
            qhit = [q for q in qset if q in f["rel"] and len(tokset(f.get("_qtext", {}).get(q, ""))) > 0]
            del qhit
            if ds and has_any(low, r"\b(summar|describe|topic|about|entity)\b"):
                return render_per_item([(s.label, f["ent"].get(s.iid, "")) for s in ds], view.instr)
            return refuse_aggregate(st)
        rng = rng_for(view.prompt, "qd")
        err = self.err_prob(view.prompt)
        results: dict[str, dict] = {}
        for q in qs:
            gold = [d for d in (f["rel"].get(q.iid) or [])]
            g_here = [s for s in ds if int(s.iid) in gold]
            r = rng_for(view.prompt, q.iid, "ans").random()
            look = self._lookalike(f, q.iid, texts[q.iid], ds)
            many_q = len(qs) > 1
            decay = max(0.25, 1.0 - 0.006 * max(0, len(qs) - 3) - 0.002 * max(0, len(ds) - 10)) if many_q else 1.0
            res = {"answer": "NOT ANSWERABLE", "used": [], "quotes": {}}
            if len(g_here) == 2 and r < (self.p_both - err) * decay:
                used = list(g_here)
                first_hop = [s for s in g_here if s in look] or g_here[:1]
                if rng_for(view.prompt, q.iid, "cite").random() > (self.p_cite_both if not quote_mode else 0.9):
                    used = first_hop[:1]
                res = {"answer": "(answer)", "used": used}
            elif len(g_here) == 1 and r < self.p_one * decay:
                res = {"answer": "(answer)", "used": g_here}
            elif not g_here and look and r < (self.p_look_quote if quote_mode else self.p_look) * decay:
                res = {"answer": "(answer)", "used": look[:1]}
            elif not g_here and r < 0.01:
                res = {"answer": "(answer)", "used": [rng.choice(ds)] if ds else []}
            results[q.label] = res
        if len(qs) == 1:
            q = qs[0]
            res = results[q.label]
            ok = res["answer"] != "NOT ANSWERABLE"
            qtext = texts[q.iid]
            named = {x for t in re.findall(r"\[(\d{1,7})\]|(?:document|doc)\s*#?\s*(\d{1,7})", low) for x in t if x}
            if has_any(low, r"\bentity\b", r"\bname of\b", r"look up", r"search for") and not \
                    has_any(low, r"which (one )?(other )?(document|passage)"):
                gold = f["rel"].get(q.iid) or []
                known = [s for s in ds if int(s.iid) in gold]
                missing = [d for d in gold if str(d) not in {s.iid for s in ds}]
                r = rng_for(view.prompt, q.iid, "ent").random()
                if known and missing and r < 0.85:
                    ent = f["ent"].get(str(missing[0])) or "unknown"
                else:
                    caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", qtext)
                    ent = caps[-1] if caps else "unknown"
                keys = _pick_json_keys(view)
                return json.dumps(fill_keys(keys or ["entity"], {"text": ent, "answer": ent})) if \
                    (keys or wants_json(view.instr)) else ent
            if named and has_any(low, r"\bother\b", r"\banother\b", r"\bsecond hop\b"):
                gold = f["rel"].get(q.iid) or []
                known_gold = any(int(s.iid) in gold for s in ds if s.label in named)
                pick = []
                if ok and known_gold:
                    pick = [s.label for s in ds if int(s.iid) in gold and s.label not in named]
                elif rng_for(view.prompt, "o2").random() < 0.05:
                    pick = [s.label for s in ds if s.label not in named][:1]
                return render_selection(pick[:1], view.instr, "[]")
            if has_any(low, r"for each document", r"each document", r"per document") and quote_mode:
                pairs = []
                for s in ds:
                    if s in res["used"] or (ok and int(s.iid) in (f["rel"].get(q.iid) or [])
                                            and rng_for(view.prompt, s.iid, "q").random() < 0.7):
                        pairs.append((s.label, _first_sentence_with(texts[s.iid], tokset(qtext))))
                    else:
                        pairs.append((s.label, "NONE"))
                return render_per_item(pairs, view.instr)
            if has_any(low, r"which (of these )?(documents|passages)", r"relevant documents", r"list (the|all)") \
                    and not has_any(low, r"\"answer\"", r"answer the question"):
                return render_selection([s.label for s in res["used"]] if ok else [], view.instr, "NONE")
            keys = _pick_json_keys(view)
            ans_text = _answer_text(f, q.iid, texts[q.iid]) if ok else "NOT ANSWERABLE"
            used = [label_value(s.label) for s in res["used"]] if ok else []
            if quote_mode:
                chain = [{"fact": _first_sentence_with(texts[s.iid], tokset(qtext)), "doc": label_value(s.label)}
                         for s in res["used"]] if ok else []
                obj = {}
                for k in keys or ["chain", "answer"]:
                    kl = k.lower()
                    obj[k] = chain if "chain" in kl or "fact" in kl or "evidence" in kl else (
                        ans_text if "answer" in kl else used)
                return json.dumps(obj, ensure_ascii=False)
            if wants_json(view.instr) or keys:
                obj = {}
                for k in keys or ["answer", "used"]:
                    kl = k.lower()
                    obj[k] = ans_text if "answer" in kl else used
                return json.dumps(obj, ensure_ascii=False)
            return f"{ans_text} (documents: {', '.join(map(str, used)) or 'none'})"
        pairs = []
        for q in qs:
            res = results[q.label]
            for s in res["used"]:
                pairs.append([label_value(q.label), label_value(s.label)])
        if has_any(low, r"for each (query|question)") and not has_any(low, r"pairs?\b"):
            return render_per_item([(q.label, [label_value(s.label) for s in results[q.label]["used"]]) for q in qs],
                                   view.instr)
        if wants_json(view.instr) or "[" in view.instr:
            return json.dumps(pairs)
        return "\n".join(f"{a}: {b}" for a, b in pairs) or "NONE"


def _answer_text(f: dict, q: str, qtext: str) -> str:
    return "Answer: " + " ".join(ctoks(qtext)[-3:])


_PAIR_HDR = re.compile(r"(?im)^[ \t>*#\-]*(?:\*\*)?(?:pair|p)\s*#?\s*(\d{1,5})\b")
_DIGIT_ASK = re.compile(r"single digit from 0|digit (?:from )?0\b.{0,40}\bto 9\b|from 0 \([^)]*\) to 9")


class ContradictionOracle(CTCOracle):
    """contradiction_iid: per-pair judgement with framing-dependent sensitivity; set search over many claims misses."""

    occurrences = True

    sens_rewrite = 0.93
    sens_contra = 0.72
    hard_frac = 0.15
    sens_hard_rewrite = 0.6
    sens_hard_contra = 0.35
    fp0 = 0.004
    fp_sim = 0.12
    batch_drop = 0.42
    batch_bad_list = 0.12
    persistent_fp = True
    persist = 0.5
    sens_strict = 0.6
    sens_hard_strict = 0.3
    fp_strict = 0.5

    def __init__(self, read, strength: str = "real"):
        super().__init__(read)
        if strength == "strong":
            self.batch_drop, self.batch_bad_list, self.persistent_fp = 0.0, 0.0, False

    @staticmethod
    def _strict(low: str) -> bool:
        return bool(re.search(r"\bare not (?:rewrites|counter-claims|contradictions)\b|\bnot counter-claims\b", low))

    def _framing(self, low: str) -> str:
        if has_any(low, r"rewr", r"reword", r"paraphras", r"altered", r"alter", r"copy", r"same statement",
                   r"same sentence", r"modified version", r"changed"):
            return "rewrite"
        return "contra"

    def _hard(self, a: str, b: str) -> bool:
        return rng_for("hardpair", min(a, b, key=int), max(a, b, key=int)).random() < self.hard_frac

    def _verdict(self, view: View, f: dict, a: Seen, b: Seen, framing: str, salt: str, n_pairs: int = 1) -> bool:
        err = self.err_prob(view.prompt)
        r = rng_for(view.prompt, a.iid, b.iid, salt).random()
        strict = self._strict(view.low)
        if str(f["partner"].get(a.iid)) == b.iid:
            hard = self._hard(a.iid, b.iid)
            if strict:
                p = self.sens_hard_strict if hard else self.sens_strict
            elif framing == "rewrite":
                p = self.sens_hard_rewrite if hard else self.sens_rewrite
            else:
                p = self.sens_hard_contra if hard else self.sens_contra
            p -= self.batch_drop * min(1.0, (n_pairs - 1) / 5)
            if self.persistent_fp:
                r = self._persist(view, a, b, r, "contra_sens")
            return r < p - err
        sim = jacc(view.prompt[a.pos:a.end], view.prompt[b.pos:b.end])
        p = self.fp0 + self.fp_sim * sim ** 1.5 + (0.01 if framing == "contra" else 0.0) + err / 5
        if strict:
            p *= self.fp_strict
        if not self.persistent_fp:
            return r < p
        return self._persist(view, a, b, r, "contra_fp") < p

    def _persist(self, view: View, a: Seen, b: Seen, r: float, kind: str) -> float:
        """Half the looks at a pair reuse its persistent judgement, half are fresh (calibrated on the real 27B)."""
        lo, hi = min(a.iid, b.iid, key=int), max(a.iid, b.iid, key=int)
        if rng_for(view.prompt, lo, hi, kind, "persist").random() < self.persist:
            return rng_for(kind, lo, hi).random()
        return r

    def _pairs_in_prompt(self, view: View) -> list[tuple[str, Seen, Seen]] | None:
        """Pairs as the root laid them out: 'Pair k' headers, or consecutive A/B items."""
        items = view.items
        hdrs = [(m.start(), m.group(1)) for m in _PAIR_HDR.finditer(view.prompt)]
        if hdrs and len(items) >= 2:
            groups: dict[str, list[Seen]] = defaultdict(list)
            order = []
            for s in items:
                lab = None
                for pos, k in hdrs:
                    if pos <= s.pos:
                        lab = k
                    else:
                        break
                if lab is None:
                    continue
                if lab not in groups:
                    order.append(lab)
                groups[lab].append(s)
            hdr_order = list(dict.fromkeys(k for _, k in hdrs))
            out = [(k, (groups[k][0] if len(groups.get(k, [])) >= 2 else None),
                    (groups[k][1] if len(groups.get(k, [])) >= 2 else None)) for k in hdr_order]
            if any(a is not None for _, a, _ in out):
                return out
        labs = [s.label.upper() for s in items]
        if len(items) >= 2 and len(items) % 2 == 0 and all(
                labs[i].startswith("A") and labs[i + 1].startswith("B") for i in range(0, len(items), 2)):
            return [(str(i // 2 + 1), items[i], items[i + 1]) for i in range(0, len(items), 2)]
        return None

    def _labels(self, low: str) -> tuple[str, str]:
        if re.search(r"\banswer(?: with)?(?: exactly one word:?)? \W*yes\W* or \W*no\b", low):
            return "YES", "NO"
        if "rewrite" in low and ("different" in low or "same" in low):
            return "REWRITE", "DIFFERENT"
        if "contradict" in low and "compatible" in low:
            return "CONTRADICT", "COMPATIBLE"
        if re.search(r"\byes\b", low) and re.search(r"\bno\b", low):
            return "YES", "NO"
        if "true" in low and "false" in low:
            return "true", "false"
        if "contradict" in low:
            return "CONTRADICT", "NOT_CONTRADICT"
        return "YES", "NO"

    def answer(self, view: View, hidden: dict, st: dict) -> str:
        f = hidden
        low = view.low
        if not view.items:
            return refuse_aggregate(st)
        framing = self._framing(low)
        pos_w, neg_w = self._labels(low)
        layout = self._pairs_in_prompt(view)
        if len(view.items) == 2 and layout is None:
            layout = [("1", view.items[0], view.items[1])]
        if layout is not None:
            bump(st, "pair_calls")
            read_ids = {s.iid for s in self.read_view(view, st)[0]}
            out = []
            for k, a, b in layout:
                if a is None or b is None or a.iid not in read_ids or b.iid not in read_ids:
                    out.append((k, neg_w))
                    continue
                v = self._verdict(view, f, a, b, framing, "pair", n_pairs=len(layout))
                out.append((k, pos_w if v else neg_w))
            if len(layout) == 1:
                w = out[0][1] if out else neg_w
                if _DIGIT_ASK.search(low):
                    return "9" if w == pos_w else "0"
                keys = _pick_json_keys(view)
                if keys:
                    return json.dumps(fill_keys(keys, {"verdict": w, "answer": w, "text": w}))
                if not wants_json(view.instr):
                    return w + (" -- the second statement reverses the finding of the first." if w == pos_w else "")
            sch = re.search(r"list of\s*(?:<?\w+>?\s*)?labels|json list|array", low)
            if sch and wants_json(view.instr):
                labs = [w for _, w in out]
                if len(labs) >= 5 and rng_for(view.prompt, "badlist").random() < self.batch_bad_list:
                    labs = labs[:-1]
                return json.dumps(labs)
            return render_per_item(out, view.instr)
        read, _ = self.read_view(view, st)
        m = len(read)
        st["agg_requests"] += 1
        if m > 40:
            bump(st, "set_search_large")
        decay = max(0.2, 1.0 - m / 900.0)
        found = []
        ids = {s.iid: s for s in read}
        sens = self.sens_rewrite if framing == "rewrite" else self.sens_contra
        for a, b in f["pairs"]:
            sa, sb = ids.get(str(a)), ids.get(str(b))
            if sa and sb:
                hard = self._hard(sa.iid, sb.iid)
                p = (sens * (0.6 if hard else 1.0)) * decay - self.err_prob(view.prompt)
                if rng_for(view.prompt, a, b, "set").random() < p:
                    found.append((sa, sb))
        rng = rng_for(view.prompt, "setfp")
        n_fp = sum(1 for _ in range(max(0, m // 40)) if rng.random() < min(0.6, m / 500.0))
        if n_fp and m >= 4:
            sample = rng.sample(read, min(m, 60))
            cands = []
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    a, b = sample[i], sample[j]
                    if str(f["partner"].get(a.iid)) == b.iid:
                        continue
                    cands.append((jacc(view.prompt[a.pos:a.end], view.prompt[b.pos:b.end]), a, b))
            cands.sort(key=lambda x: -x[0])
            for _sim, a, b in cands[:n_fp]:
                found.append((a, b))
        pairs = [[label_value(a.label), label_value(b.label)] for a, b in found]
        if wants_json(view.instr) or "[[" in view.instr or "[" in view.instr:
            return json.dumps(pairs)
        return "\n".join(f"{a} and {b}" for a, b in pairs) or "No contradicting pairs found."


_FIELDS = ["History", "Sports", "Music", "Politics", "Geography", "Film and television", "Science", "Biography",
           "Military", "Religion", "Literature", "Business", "Transportation", "Architecture"]


class OutlierOracle(CTCOracle):
    """outlier / outlier_fixedM: chunks of Wikipedia articles; the outliers are the one 3-chunk article."""

    frag_alone = 0.33
    frag_with = 0.07
    same_ok = 0.93
    near_fp = 0.03

    def _label(self, view: View, f: dict, s: Seen, counts: Counter, broad: bool) -> str:
        a = f["art"][s.iid]
        title = f["title"][str(a)]
        if broad:
            return _FIELDS[rng_for("field", title).randrange(len(_FIELDS))]
        p = self.frag_with if counts[a] >= 2 else self.frag_alone
        p = p + self.err_prob(view.prompt)
        r = rng_for(view.prompt, s.iid, "frag").random()
        if r < p:
            return _fragment_label(view.prompt[s.pos:s.end], title)
        return title

    def answer(self, view: View, hidden: dict, st: dict) -> str:
        f = hidden
        low = view.low
        from .fixedm_protocol import protocol_answer
        proto = protocol_answer(self, view, hidden, st, _fragment_label, bump)
        if proto is not None:
            return proto
        from .outlier_protocol import protocol_answer as outlier_member
        proto = outlier_member(self, view, hidden, st, bump)
        if proto is not None:
            return proto
        if "join directly at a cut" in view.prompt.lower() and "art" in f and "chunk" in f:
            left = re.search(r"END Document \[(\d+)\]", view.prompt)
            right = re.search(r"START Document \[(\d+)\]", view.prompt)
            if left and right and left.group(1) in f["art"] and right.group(1) in f["art"]:
                a, b = left.group(1), right.group(1)
                truth = (f["art"][a] == f["art"][b] and f["chunk"][b] == f["chunk"][a] + 1)
                r = rng_for(view.prompt, "pair-seam").random()
                same = r < ((0.94 - self.err_prob(view.prompt)) if truth else 0.03)
                bump(st, "pair_seam_calls")
                return "YES" if same else "NO"
        if "predict the first one to four words" in low and "art" in f and "chunk" in f:
            marks = list(re.finditer(r"(?m)^CASE (\d+):\s*", view.prompt))
            texts = {str(i): str(t) for i, t in f.get("items") or []}
            tails = {" ".join(t.split())[-80:]: i for i, t in texts.items()}
            by_source = {(f["art"][i], f["chunk"][i]): i for i in texts if i in f["art"]}
            answers = {}
            for k, mark in enumerate(marks):
                end = marks[k + 1].start() if k + 1 < len(marks) else len(view.prompt)
                shown = " ".join(view.prompt[mark.end():end].split())[-80:]
                iid = tails.get(shown)
                if iid is None:
                    bump(st, "boundary_completion_unmatched_cases")
                next_id = by_source.get((f["art"][iid], f["chunk"][iid] + 1)) if iid else None
                sample = rng_for(view.prompt, mark.group(1), "boundary-completion")
                if next_id and sample.random() < 0.65:
                    next_words = re.findall(r"[A-Za-z0-9]+", texts[next_id])[:2]
                    answers[mark.group(1)] = [" ".join(next_words)] if next_words else []
                elif sample.random() < 0.03 and texts:
                    other = sample.choice(list(texts.values()))
                    answers[mark.group(1)] = [" ".join(re.findall(r"[A-Za-z0-9]+", other)[:2])]
                else:
                    answers[mark.group(1)] = ["unknown continuation"]
            bump(st, "boundary_completion_calls")
            return json.dumps(answers)
        if not view.items:
            return refuse_aggregate(st)
        read, _ = self.read_view(view, st)
        counts = Counter(f["art"][s.iid] for s in read)
        if "each case is independent" in low and '"all_same"' in view.instr and \
           "same exact wikipedia article" in low:
            marks = list(re.finditer(r"(?m)^CASE (\d+)\s*$", view.prompt))
            heads = {" ".join(str(t).split())[:80]: str(i) for i, t in f.get("items") or []}
            verdicts = {}
            for k, mark in enumerate(marks):
                end = marks[k + 1].start() if k + 1 < len(marks) else len(view.prompt)
                case = view.prompt[mark.end():end]
                starts = list(re.finditer(r"(?m)^G[123]:\s*", case))
                groups = []
                for n, start in enumerate(starts):
                    stop = starts[n + 1].start() if n + 1 < len(starts) else len(case)
                    head = " ".join(case[start.end():stop].split())[:80]
                    groups.append(heads.get(head))
                if len(groups) != 3 or any(i is None for i in groups):
                    bump(st, "batch_internal_unmatched_cases")
                truth = (len(groups) == 3 and all(i in f["art"] for i in groups) and
                         len({f["art"][i] for i in groups}) == 1)
                same = rng_for(view.prompt, mark.group(1), "batch-internal").random() < \
                    (0.92 if truth else 0.06)
                verdicts[mark.group(1)] = {
                    "all_same": "YES" if same else "NO",
                    "evidence": "same article" if same else "different article",
                }
            bump(st, "batch_internal_calls")
            return json.dumps(verdicts)
        if "each case is independent" in low and "same exact wikipedia article" in low:
            marks = list(re.finditer(r"(?m)^CASE (\d+)\s*$", view.prompt))
            heads = {" ".join(str(t).split())[:80]: str(i) for i, t in f.get("items") or []}
            verdicts = {}
            for k, mark in enumerate(marks):
                end = marks[k + 1].start() if k + 1 < len(marks) else len(view.prompt)
                case = view.prompt[mark.end():end]
                starts = list(re.finditer(r"(?m)^(G[123]|O1):\s*", case))
                matched = []
                for n, start in enumerate(starts):
                    stop = starts[n + 1].start() if n + 1 < len(starts) else len(case)
                    head = " ".join(case[start.end():stop].split())[:80]
                    matched.append((start.group(1), heads.get(head)))
                groups = [iid for label, iid in matched if label.startswith("G") and iid is not None]
                options = [iid for label, iid in matched if label == "O1" and iid is not None]
                ids = groups + options
                if len(groups) != 3 or len(options) != 1:
                    bump(st, "batch_confirm_unmatched_cases")
                truth = (len(groups) == 3 and len(options) == 1 and
                         all(i in f["art"] for i in ids) and
                         len({f["art"][i] for i in ids}) == 1)
                same = rng_for(view.prompt, mark.group(1), "batch-confirm").random() < \
                    (0.73 if truth else 0.01)
                verdicts[mark.group(1)] = "YES" if same else "NO"
            bump(st, "batch_confirm_calls")
            return json.dumps(verdicts)
        if "directly continues" in low and '"continues"' in view.instr:
            groups = [s for s in read if s.label.upper().startswith("G")]
            options = [s for s in read if s.label.upper().startswith("O")]
            named = []
            for s in options:
                truth = any(f["art"][s.iid] == f["art"][g.iid]
                            and abs(f["chunk"][s.iid] - f["chunk"][g.iid]) == 1 for g in groups)
                r = rng_for(view.prompt, s.iid, "seam").random()
                if (truth and r < 0.85 - self.err_prob(view.prompt)) or (not truth and r < 0.04):
                    named.append(s.label)
            bump(st, "seam_calls")
            return json.dumps({"continues": named})
        if "direct boundary" in low and '"same"' in view.instr:
            groups = [s for s in read if s.label.upper().startswith("G")]
            options = [s for s in read if s.label.upper().startswith("O")]
            truth = any(f["art"][s.iid] == f["art"][g.iid]
                        and abs(f["chunk"][s.iid] - f["chunk"][g.iid]) == 1
                        for s in options for g in groups)
            r = rng_for(view.prompt, "seam-confirm").random()
            yes = r < ((0.90 - self.err_prob(view.prompt)) if truth else 0.05)
            bump(st, "seam_confirm_calls")
            return json.dumps({"same": "YES" if yes else "NO",
                               "connection": "adjacent chunk boundary" if yes else "no direct boundary"})
        if "all three" in low and '"all_same"' in view.instr:
            groups = [s for s in read if s.label.upper().startswith("G")]
            arts = [f["art"][s.iid] for s in groups]
            truth = len(groups) == 3 and len(set(arts)) == 1
            r = rng_for(view.prompt, "internal-article").random()
            same = r < ((0.92 - self.err_prob(view.prompt)) if truth else 0.06)
            different = "none"
            if not same and groups:
                counts_art = Counter(arts)
                minority = next((k for k, a in enumerate(arts) if counts_art[a] == 1), 0)
                different = groups[minority].label
            bump(st, "internal_article_calls")
            return json.dumps({"all_same": "YES" if same else "NO", "different": different,
                               "evidence": "one passage differs in article identity" if not same
                               else "all three passages share one article"})
        if '"parent_page"' in view.instr and '"same"' in view.instr and "fourth chunk" in low:
            groups = [s for s in read if s.label.upper().startswith("G")]
            options = [s for s in read if s not in groups]
            truth = (len(groups) == 3 and len(options) == 1 and
                     len({f["art"][s.iid] for s in groups + options}) == 1)
            r = rng_for(view.prompt, "parent-page-confirm").random()
            same = r < (max(0.0, 0.82 - self.err_prob(view.prompt)) if truth else 0.025)
            group_art = Counter(f["art"][s.iid] for s in groups).most_common(1)[0][0] if groups else None
            page = f["title"].get(str(group_art), "none") if same and group_art is not None else "none"
            bump(st, "parent_page_confirm_calls")
            return json.dumps({"parent_page": page, "same": "YES" if same else "NO",
                               "evidence": "same source page" if same else "different source page"})
        if '"same"' in view.instr and \
           (("was the candidate" in low and "all three a chunks" in low) or
            "were o1 and all three g chunks" in low):
            groups = [s for s in read if s.label.upper().startswith("G")]
            options = [s for s in read if s not in groups]
            truth = (len(groups) == 3 and len(options) == 1 and
                     len({f["art"][s.iid] for s in groups + options}) == 1)
            r = rng_for(view.prompt, "confirm-all-three").random()
            same = r < (0.98 if truth else 0.005)
            bump(st, "confirm_article_calls")
            return json.dumps({"same": "YES" if same else "NO",
                               "evidence": "one Wikipedia article" if same else "different Wikipedia articles"})
        prefixes = [re.match(r"[A-Za-z]*", s.label).group(0).upper() for s in read]
        kinds = list(dict.fromkeys(p for p in prefixes if p))
        same_q = has_any(low, r"same (wikipedia )?article", r"same source", r"same (document|page|topic|subject)",
                         r"cut from", r"belong together", r"from one article")
        if same_q:
            bump(st, "same_article_calls")
        if same_q and len(kinds) >= 2:
            grp = [s for s, p in zip(read, prefixes, strict=True) if p == kinds[0]]
            oth = [s for s, p in zip(read, prefixes, strict=True) if p != kinds[0]]
            ga = Counter(f["art"][s.iid] for s in grp).most_common(1)[0][0]
            gtext = " ".join(view.prompt[s.pos:s.end] for s in grp)
            named = []
            for s in oth:
                r = rng_for(view.prompt, s.iid, "near").random()
                same = f["art"][s.iid] == ga
                fp = self.near_fp * min(1.0, 8 * jacc(gtext, view.prompt[s.pos:s.end]))
                ok = self.same_ok - (0.08 if f.get("fam") == "outlier_fixedM" else 0.0)
                if (same and r < ok - self.err_prob(view.prompt)) or (not same and r < fp):
                    named.append(s.label)
            if re.search(r'"same"\s*:', view.instr):
                subj = {s.label: f["title"][str(f["art"][s.iid])] for s in oth}
                return json.dumps({"group": f["title"][str(ga)], "others": subj, "same": named}, ensure_ascii=False)
            return render_selection(named, view.instr, "[]")
        if same_q and len(read) <= 8 and has_any(low, r"\b(were|are|is|do|did)\b.*\?"):
            arts = {f["art"][s.iid] for s in read}
            r = rng_for(view.prompt, "same").random()
            truth = len(arts) == 1
            if truth:
                ans = r < self.same_ok
            else:
                sim = max(jacc(view.prompt[a.pos:a.end], view.prompt[b.pos:b.end]) for a in read for b in read
                          if a.iid != b.iid and f["art"][a.iid] != f["art"][b.iid])
                ans = r < (0.10 if f.get("fam") == "outlier_fixedM" else 0.12) + 0.4 * sim
            title = f["title"][str(next(iter(arts)))] if truth else "several different articles"
            keys = _pick_json_keys(view)
            if wants_json(view.instr) or keys:
                obj = {}
                for k in keys or ["same", "article"]:
                    kl = k.lower()
                    obj[k] = ("YES" if ans else "NO") if ("same" in kl or "answer" in kl or "verdict" in kl) \
                        else (title if ans else "different articles")
                return json.dumps(obj)
            return ("YES" if ans else "NO") + (f" -- all from the article on {title}." if ans else
                                               " -- they come from different articles.")
        grouped = _pieces_of(view)
        if grouped and has_any(low, r"\btitle\b", r"\barticle\b", r"\btopic\b", r"\bsubject\b"):
            bump(st, "title_calls")
            out = []
            for lab, members in grouped:
                arts = Counter(f["art"][s.iid] for s in members if s.iid in {x.iid for x in read})
                if not arts:
                    continue
                a, k = arts.most_common(1)[0]
                title = f["title"][str(a)]
                p = (self.frag_with / 2 if k >= 2 else (self.frag_with if counts[a] >= 2 else self.frag_alone))
                if rng_for(view.prompt, lab, "gfrag").random() < p + self.err_prob(view.prompt):
                    src = next(s for s in members if f["art"][s.iid] == a)
                    title = _fragment_label(view.prompt[src.pos:src.end], title)
                out.append((lab, title))
            return render_per_item(out, view.instr)
        if has_any(low, r"outlier", r"different topic", r"does not belong", r"don't belong", r"odd one",
                   r"stand out", r"unrelated to the rest", r"different from the rest", r"minority"):
            st["agg_requests"] += 1
            return self._outliers(view, f, read, counts)
        broad = has_any(low, r"broad", r"general (topic|category|field|area|domain)", r"\bdomain\b", r"\bfield\b",
                        r"\bcategor")
        if has_any(low, r"\b(group|cluster|partition)\b") and not has_any(low, r"\b(label|topic|title|subject)\b"):
            st["agg_requests"] += 1
            return self._cluster(view, f, read, counts)
        bump(st, "title_calls")
        pairs = [(s.label, self._label(view, f, s, counts, broad)) for s in read]
        return render_per_item(pairs, view.instr)

    def _outliers(self, view: View, f: dict, read: list[Seen], counts: Counter) -> str:
        n = len(read)
        rng = rng_for(view.prompt, "outl")
        arts = counts.most_common()
        small = min(c for _, c in arts)
        minority = [a for a, c in arts if c == small]
        simple = len(arts) <= 4 and arts[0][1] >= 3 * small and len(minority) == 1
        if simple:
            p_ok = 0.95 if n <= 60 else (0.6 if n <= 120 else 0.3)
            if rng.random() < p_ok:
                pick = [s for s in read if f["art"][s.iid] == minority[0]]
            else:
                pick = [s for s in read if f["art"][s.iid] != arts[0][0]]
        elif n <= self.read.agg_capacity:
            pick = [s for s in read if f["art"][s.iid] in minority] if len(minority) < len(arts) else []
        else:
            k = rng.randint(2, min(8, max(2, n // 6)))
            pick = rng.sample(read, min(k, n))
        labs = [s.label for s in pick]
        if wants_json(view.instr):
            return json.dumps([label_value(lb) for lb in labs])
        maj = f["title"][str(arts[0][0])] if simple else "a mix of topics"
        return (f"Most passages are about {maj}; the outliers are: "
                + ", ".join(f"[{lb}]" if lb.isdigit() else lb for lb in labs))

    def _cluster(self, view: View, f: dict, read: list[Seen], counts: Counter) -> str:
        groups: dict[str, list] = defaultdict(list)
        for s in read:
            lab = self._label(view, f, s, counts, False) if len(read) > 30 else f["title"][str(f["art"][s.iid])]
            groups[lab].append(label_value(s.label))
        if wants_json(view.instr):
            return json.dumps([{"topic": k, "ids": v} for k, v in groups.items()])
        return "\n".join(f"{k}: {v}" for k, v in groups.items())


def _fragment_label(text: str, title: str) -> str:
    caps = re.findall(r"\b[A-Z][a-z]+(?:\s+(?:of\s+|the\s+|de\s+)?[A-Z][a-z]+)+\b", text)
    tl = title.lower()
    caps = [c for c in caps if c.lower() not in tl and tl not in c.lower()]
    if caps:
        return Counter(caps).most_common(1)[0][0]
    words = [w for w in re.findall(r"[A-Za-z]{5,}", text) if w.lower() not in STOP]
    return " ".join(words[:3]).title() if words else "General topic"


_SYN = {
    "Computer science": ["Computing", "Computer Science", "Informatics", "CS / machine learning"],
    "Medicine": ["Clinical medicine", "Medical science", "Health sciences", "Medicine and health"],
    "Biology": ["Life sciences", "Biological sciences", "Biology (life science)"],
    "Chemistry": ["Chemical sciences", "Chemistry (general)", "Chemical research"],
    "Materials science": ["Materials engineering", "Materials Science", "Material science"],
    "Physics": ["Physical sciences", "Physics (general)", "Applied physics"],
    "Psychology": ["Behavioral science", "Psychological science", "Psychology research"],
    "Environmental science": ["Environmental studies", "Ecology and environment", "Environmental Science"],
    "Business": ["Management", "Business and management", "Business studies"],
    "Economics": ["Economic science", "Economics and finance", "Economic research"],
    "Political science": ["Politics", "Government and policy", "Political Science"],
    "Sociology": ["Social science", "Social sciences", "Sociology research"],
    "Geography": ["Geographic studies", "Human geography", "Geography (spatial)"],
    "Geology": ["Earth sciences", "Geoscience", "Geological sciences"],
    "Engineering": ["Engineering (general)", "Technology and engineering", "Applied engineering"],
    "Mathematics": ["Math", "Mathematical sciences", "Applied mathematics"],
    "Art": ["Arts", "Fine arts", "Art and design"],
    "History": ["Historical studies", "History (general)", "Historical research"],
    "Philosophy": ["Philosophical studies", "Philosophy (general)", "Ethics and philosophy"],
}
_CONFUSE = {"Materials science": "Chemistry", "Chemistry": "Materials science", "Biology": "Medicine",
            "Medicine": "Biology", "Business": "Economics", "Economics": "Business", "Environmental science": "Geology",
            "Political science": "Sociology", "Sociology": "Political science", "Geography": "Environmental science",
            "Psychology": "Sociology", "Computer science": "Engineering", "Engineering": "Computer science",
            "Physics": "Materials science", "Geology": "Environmental science", "Mathematics": "Computer science"}


def label_variants(label: str) -> list[str]:
    if label in _SYN:
        return _SYN[label]
    base = re.sub(r"\s*\(.*?\)\s*", " ", label).strip() or label
    return [f"{base} research", f"Studies of {base.lower()}", f"{base} (topic)", base.lower()]


def canon_of(variant: str, labels: set[str]) -> str | None:
    v = variant.strip().lower()
    for lab in labels:
        if v == lab.lower() or v in (x.lower() for x in label_variants(lab)):
            return lab
    return None


class GroupingOracle(CTCOracle):
    """grouping: per-paper field labels (free-form variants or a fixed list), pairwise same-topic, label merging."""

    partial = 0.1

    p_exact_free = 0.6
    err_fixed = 0.07
    err_deep = 0.12
    same_ok = 0.9

    def _options(self, view: View, labels: set[str]) -> list[str]:
        """Field names the instruction offers (numbered / bulleted / quoted / comma lists)."""
        instr = view.instr
        opts = []
        for m in re.finditer(r"(?m)^\s*(?:\d{1,3}[.):]|[-*•]|\(\d{1,3}\))\s*([^\n:]{2,80}?)(?:\s*[:\-–]\s.*)?$", instr):
            opts.append(m.group(1).strip().strip('"'))
        if re.search(r"(?i)\b(categor|options?|one of|choose from|labels? (?:to use|available)|pick from)", instr):
            opts += [q for q in re.findall(r'"([^"\n]{2,60})"', instr) if "<" not in q and ">" not in q]
        opts = [o for o in opts if "<" not in o and ">" not in o and not re.fullmatch(r"(?i)x|id|label|name", o)]
        if len(opts) < 2:
            m = re.search(r"(?:one of|options?|categories|choose from|labels?)[^:\n]{0,40}:\s*([^\n]+)", instr, re.I)
            if m:
                opts += [x.strip(" .") for x in re.split(r",|;|\|", m.group(1)) if x.strip()]
        known = []
        low_labels = {lb.lower(): lb for lb in labels}
        rule = re.compile(r"(?i)\b(must|should|do not|don't|never|reply|return|respond|output|each|every|exactly|only|"
                          r"json|format|group|cluster|categor(y|ies)|paper|document|id)\b")
        for o in opts:
            if o.lower() in low_labels or canon_of(o, labels) or (len(o.split()) <= 6 and not rule.search(o)):
                known.append(o)
        return list(dict.fromkeys(known))

    def answer(self, view: View, hidden: dict, st: dict) -> str:
        f = hidden
        low = view.low
        labels = set(f["label"].values())
        if not view.items:
            if has_any(low, r"merge", r"synonym", r"canonical", r"map (each|these)", r"consolidat", r"normali"):
                return self._merge(view, f, labels, st)
            return refuse_aggregate(st)
        read, _ = self.read_view(view, st)
        deep = f["level"] >= 2
        err = self.err_prob(view.prompt) + (self.err_deep if deep else self.err_fixed)
        if len(read) == 2 and has_any(low, r"same (topic|field|category|group|cluster|research|area|subject|concept)",
                                      r"belong (to|in) the same", r"should .* be grouped"):
            a, b = read
            truth = f["label"][a.iid] == f["label"][b.iid]
            r = rng_for(view.prompt, "same").random()
            ans = truth if r < self.same_ok else not truth
            return "YES" if ans else "NO"
        if has_any(low, r"\b(cluster|partition|group)\b.*\b(these|the following|them)\b") and \
                not has_any(low, r"\blabel\b", r"for each (paper|abstract|document)"):
            st["agg_requests"] += 1
            return self._cluster(view, f, read, err)
        opts = self._options(view, labels)
        bump(st, "fixed_list_calls" if len(opts) >= 2 else "free_label_calls")
        out = []
        for s in read:
            gold = f["label"][s.iid]
            r = rng_for(view.prompt, s.iid, "lab").random()
            if len(opts) >= 2:
                canon = {o: (canon_of(o, labels) or o) for o in opts}
                hit = [o for o in opts if canon[o] == gold]
                if hit and r >= err:
                    val = hit[0]
                else:
                    conf = _CONFUSE.get(gold)
                    alt = [o for o in opts if canon[o] == conf] or [o for o in opts if canon[o] != gold] or opts
                    val = alt[rng_for(view.prompt, s.iid, "alt").randrange(len(alt))]
                num = re.search(r"(?:reply|answer|give|return)[^.\n]{0,40}\b(number|index)\b", low)
                out.append((s.label, (opts.index(val) + 1) if num else val))
            else:
                if r < err:
                    gold = _CONFUSE.get(gold) or gold
                v = gold if rng_for(view.prompt, s.iid, "var").random() < self.p_exact_free else \
                    rng_for(view.prompt, s.iid, "pick").choice(label_variants(gold))
                if has_any(low, r"two[- ]level", r"discipline.*subfield", r"field.*subfield", r"broad.*specific"):
                    out.append((s.label, [_broad(gold), v]))
                else:
                    out.append((s.label, v))
        return render_per_item(out, view.instr)

    def _merge(self, view: View, f: dict, labels: set[str], st: dict) -> str:
        bump(st, "merge_calls")
        cand = re.findall(r'"([^"\n]{2,80})"', view.instr) or \
            [x.strip(" -*•\t") for x in view.instr.split("\n") if 2 <= len(x.strip(" -*•\t")) <= 80]
        mapping = {}
        for c in cand[:80]:
            k = re.sub(r"\s*\(\d+\)\s*$|\s*[:\-]\s*\d+\s*$", "", c).strip()
            can = canon_of(k, labels)
            if can:
                mapping[k] = can
        return json.dumps(mapping, ensure_ascii=False) if wants_json(view.instr) else \
            "\n".join(f"{k} -> {v}" for k, v in mapping.items())

    def _cluster(self, view: View, f: dict, read: list[Seen], err: float) -> str:
        n = len(read)
        rng = rng_for(view.prompt, "clu")
        labs = sorted({f["label"][s.iid] for s in read})
        groups: dict[str, list] = defaultdict(list)
        for s in read:
            g = f["label"][s.iid]
            if n > self.read.agg_capacity and rng.random() < min(0.6, err + n / 300):
                g = rng.choice(labs)
            groups[g].append(label_value(s.label))
        named = {k: rng_for("gname", k).choice(label_variants(k)) for k in groups}
        if wants_json(view.instr):
            return json.dumps({"groups": [{"label": named[k], "doc_ids": v} for k, v in groups.items()]})
        return "\n".join(f"{named[k]}: {v}" for k, v in groups.items())


def _broad(label: str) -> str:
    return label if label in _SYN else _FIELDS[rng_for("broad", label).randrange(len(_FIELDS))]


SCORE_PARAMS = {
    "u": {"true": 4.8, "-1": 3.1, "2": 3.4, "3-5": 2.1, "near": 2.1, "far": 0.0, "seam": 2.4},
    "u_edge": {"true": 4.4, "-1": 1.8, "2": 2.6, "3-5": 2.4, "near": 1.2, "far": 0.0, "seam": 2.5},
    "persist": 0.1, "none": 0.1, "runner_up": 0.8, "low_hi": 0.04,
}
SCORE_PARAMS_STRONG = {
    "u": {"true": 6.0, "-1": 1.0, "2": 1.0, "3-5": 0.5, "near": 0.5, "far": 0.0, "seam": 2.0},
    "u_edge": {"true": 6.0, "-1": 0.5, "2": 1.0, "3-5": 0.5, "near": 0.5, "far": 0.0, "seam": 2.0},
    "persist": 0.0, "none": 0.02, "runner_up": 0.3, "low_hi": 0.0,
}
_WIN = ((10, 0.35), (9, 0.5), (8, 0.15))
_LOW = ((0, 0.45), (1, 0.32), (2, 0.11), (3, 0.04), (4, 0.03), (5, 0.02), (6, 0.02), (7, 0.01))


def score_bucket(gap: int, seam: bool = False) -> str:
    if gap == 1:
        return "seam" if seam else "true"
    return "-1" if gap == -1 else "2" if gap == 2 else "3-5" if 3 <= gap <= 5 else "near" if abs(gap) <= 5 else "far"


def _draw(r, table) -> int:
    u, acc = r.random(), 0.0
    for v, w in table:
        acc += w
        if u < acc:
            return v
    return table[-1][0]


def score_candidates(buckets: list[str], pairs: list, prompt: str, params: dict, edge: bool) -> list[int]:
    """0-10 scores for one prompt's candidates; ``pairs`` keys the persistent part of the noise."""
    u = params["u_edge" if edge else "u"]
    z = []
    for b, pr in zip(buckets, pairs, strict=True):
        look = rng_for(prompt, *pr, "look")
        r = rng_for(*pr, "pair") if look.random() < params["persist"] else look
        z.append(u[b] - math.log(-math.log(max(1e-12, r.random()))))
    r = rng_for(prompt, "prompt")
    if r.random() < params["none"]:
        return [_draw(rng_for(prompt, k, "low"), _LOW) for k in range(len(z))]
    order = sorted(range(len(z)), key=lambda k: -z[k])
    out = [0] * len(z)
    out[order[0]] = _draw(r, _WIN)
    for rank, k in enumerate(order[1:], 1):
        rk = rng_for(prompt, k, "low")
        close = z[order[0]] - z[k]
        hi = (rank == 1 and close < 1.0 and rk.random() < params["runner_up"]) or rk.random() < params["low_hi"]
        out[k] = out[order[0]] - 1 - (rk.random() < 0.4) if hi else min(out[order[0]] - 1, _draw(rk, _LOW))
    return out


class ReorderOracle(CTCOracle):
    """reorder: shuffled consecutive passages of one book; local successor judgements, poor global orders."""

    partial = 0.15
    occurrences = True

    def __init__(self, read: ReadModel | None = None, strength: str = "real"):
        super().__init__(read)
        if strength not in ("real", "strong"):
            raise ValueError(f"reorder oracle strength must be real | strong, got {strength!r}")
        self.strength = strength

    @property
    def real(self) -> bool:
        return self.strength == "real"

    def respond(self, prompt: str, hidden: dict, st: dict) -> str:
        from .oracle_base import build_view, texts_of
        texts = texts_of(hidden)
        view = build_view(prompt, hidden["_index"], self.read.min_cov, texts=texts, partial=self.partial,
                          occurrences=True)
        _augment_partials(view, texts)
        from .oracle_base import note_call
        full = {x.iid for x in view.items if x.full}
        note_call(st, prompt, len(full), len(texts))
        reads = st.setdefault("read_ids", {})
        for iid in full:
            reads[iid] = reads.get(iid, 0) + 1
        st["items_read"] = len(reads)
        return self.chat(prompt, self.answer(view, hidden, st))

    def acc(self, n_targets: int, n_cand: int, full: bool, backward: bool = False) -> float:
        if self.real:
            if full:
                base = 0.83 if n_cand <= 5 else 0.6 if n_cand <= 20 else 0.45 if n_cand <= 60 else 0.3
            else:
                base = 0.61 if n_cand <= 5 else 0.48 if n_cand <= 20 else 0.37 if n_cand <= 60 else 0.27
            base *= 1.0 if n_targets <= 8 else 0.85 if n_targets <= 16 else 0.6
            if backward and not full:
                base *= 1.0 if n_cand <= 20 else 0.6 if n_cand <= 60 else 0.3
            return min(0.95, base)
        base = 0.97 if n_targets <= 2 else 0.95 if n_targets <= 8 else 0.85 if n_targets <= 16 else 0.6
        if n_cand > 120:
            base -= 0.2
        elif n_cand > 30:
            base -= 0.02
        return base * (1.0 if full else 0.97)

    def answer(self, view: View, hidden: dict, st: dict) -> str:
        f = hidden
        low = view.low
        if not view.items:
            return refuse_aggregate(st)
        pos = {k: int(v) for k, v in f["pos"].items()}
        seams = set(f["seams"])

        items = view.items
        if has_any(low, r"\bpieces?\b", r"\bchunks?\b", r"\bsegments?\b", r"\bblocks?\b", r"\bchains?\b",
                   r"\bgroups?\b"):
            pieces = _pieces_of(view)
            if pieces and len(pieces) >= 2:
                bump(st, "piece_calls")
                return self._order_pieces(view, f, pieces, pos, st)
        if has_any(low, r"\b(rate|score)\b") and has_any(low, r"\b0\s*(?:-|to)\s*10\b") and \
                has_any(low, r"\bafter\b", r"continu", r"\bnext\b", r"follows?\b", r"successor"):
            bump(st, "score_calls")
            return self._scores(view, pos, seams)
        backward = has_any(low, r"\bbefore\b", r"\bprecede", r"\bprevious", r"\bpredecessor")
        forward = has_any(low, r"\bnext\b", r"\bafter\b", r"follows?\b", r"continu", r"successor") or backward
        if (forward or backward) and not has_any(low, r"original order", r"full order", r"order of all",
                                                 r"put .* in order", r"permutation"):
            bump(st, "link_calls")
            return self._links(view, f, pos, seams, backward, st)
        if len(items) == 2 and has_any(low, r"earlier", r"\bfirst\b", r"comes before", r"which .* precedes"):
            a, b = items
            truth = a if pos[a.iid] < pos[b.iid] else b
            gap = abs(pos[a.iid] - pos[b.iid])
            if self.real:
                p = 0.62 if gap == 1 else 0.7 if gap <= 5 else 0.55 if gap <= 15 else 0.5
            else:
                p = 0.85 if gap == 1 else 0.92
            ans = truth if rng_for(view.prompt, "pw").random() < p else (b if truth is a else a)
            return render_selection([ans.label], view.instr) if wants_json(view.instr) else ans.label
        if has_any(low, r"order", r"sequence", r"arrange", r"permutation", r"chronolog"):
            st["agg_requests"] += 1
            if len(items) > 16:
                bump(st, "order_large")
            return self._order(view, f, pos, st)
        if has_any(low, r"open", r"begin", r"start", r"first passage"):
            n = len(items)
            truth = min(items, key=lambda s: pos[s.iid])
            p = min(0.9, 1.5 / n) if self.real else 0.85 if n <= 10 else 0.6 if n <= 30 else 0.3
            ans = truth if rng_for(view.prompt, "open").random() < p else rng_for(view.prompt, "o2").choice(items)
            return render_selection([ans.label], view.instr) if wants_json(view.instr) else ans.label
        pairs = [(s.label, _describe(view.prompt[s.pos:s.end])) for s in items]
        return render_per_item(pairs, view.instr)

    def _scores(self, view: View, pos: dict, seams: set) -> str:
        items = view.items
        tgt = next((s for s in items if s.label.upper() in ("A", "PASSAGE A")), None) or items[0]
        cands = [s for s in items if s.iid != tgt.iid and s.label and s.label.upper() not in ("A", "PASSAGE A")]
        cands = list({s.label: s for s in cands}.values())
        if not cands:
            return "{}"
        edge = not tgt.full or not all(c.full for c in cands) or has_any(view.low, r"\bopenings?\b", r"end of passage")
        buckets = [score_bucket(pos[c.iid] - pos[tgt.iid], pos[c.iid] in seams) for c in cands]
        params = SCORE_PARAMS if self.real else SCORE_PARAMS_STRONG
        vals = score_candidates(buckets, [(tgt.iid, c.iid) for c in cands], view.prompt, params, edge)
        return json.dumps({label_value(c.label): v for c, v in zip(cands, vals, strict=True)})

    def _links(self, view: View, f: dict, pos: dict, seams: set, backward: bool, st: dict) -> str:
        items = view.items
        prefixes = [re.match(r"[A-Za-z]*", s.label).group(0).upper() for s in items]
        tgt_pref = "B" if backward and "B" in prefixes else "A"
        targets = [s for s, p in zip(items, prefixes, strict=True) if p == tgt_pref and s.full]
        if not targets:
            targets = [s for s in items if s.full] if len(items) <= 2 else \
                [s for s in items if s.full and re.match(r"^\d+$", s.label) is not None]
            if len(items) == 2:
                a, b = items
                truth = (pos[b.iid] == pos[a.iid] + 1) if not backward else (pos[a.iid] == pos[b.iid] + 1)
                seam = (pos[b.iid] in seams) if not backward else (pos[a.iid] in seams)
                r = rng_for(view.prompt, "pair").random()
                if self.real:
                    ans = r < ((0.2 if seam else 0.45) if truth else 0.17)
                else:
                    p = 0.9 if not seam else 0.45
                    ans = truth if r < p else not truth
                return "YES" if ans else "NO"
        cands = [s for s in items if s not in targets] or items
        n_c = len(cands)
        out = []
        by_pos = {pos[s.iid]: s for s in cands}
        for k_t, t in enumerate(targets):
            if k_t >= 12 and rng_for(view.prompt, t.iid, "skipT").random() < 0.5:
                st["items_skipped"] += 1
                continue
            p_t = pos[t.iid]
            true_pos = p_t - 1 if backward else p_t + 1
            seam = (p_t in seams) if backward else (true_pos in seams)
            cand = by_pos.get(true_pos)
            if cand is t:
                cand = None
            r = rng_for(view.prompt, t.iid, "link")
            full = all(c.full for c in cands)
            if cand is not None and self.real:
                acc = self.acc(len(targets), n_c, cand.full and full, backward)
                others = [c for c in cands if c is not cand and c is not t]
                u = r.random()
                if seam:
                    ans = cand.label if u < 0.12 else ("NONE" if u < 0.35 or not others else r.choice(others).label)
                elif u < acc:
                    ans = cand.label
                else:
                    ans = "NONE" if (r.random() < 0.1 or not others) else r.choice(others).label
            elif cand is not None:
                acc = self.acc(len(targets), n_c, cand.full and full) - self.err_prob(view.prompt)
                if seam:
                    u = r.random()
                    ans = cand.label if u < 0.4 else ("NONE" if u < 0.85 else r.choice(cands).label)
                else:
                    ans = cand.label if r.random() < acc else (r.choice([c for c in cands if c is not cand]).label
                                                              if n_c > 1 and r.random() < 0.7 else "NONE")
            else:
                p_none = (0.23 if self.real else 0.88) if (n_c <= 5 and full) else (0.07 if self.real else 0.7)
                ans = "NONE" if r.random() < p_none else r.choice(cands).label
            if ans == t.label:
                ans = "NONE"
            out.append((t.label, label_value(ans) if ans != "NONE" else "NONE"))
        if wants_json(view.instr) and re.search(r"null", view.instr):
            out = [(k, None if v == "NONE" else v) for k, v in out]
        keys = _pick_json_keys(view)
        if len(targets) == 1 and keys and not any(k.upper().startswith(("A1", "B1")) for k in keys):
            v = out[0][1] if out else None
            return json.dumps(fill_keys(keys, {"id": v, "answer": v, "text": v, "verdict": v}))
        return render_per_item(out, view.instr)

    def _order(self, view: View, f: dict, pos: dict, st: dict) -> str:
        items = view.items
        n = len(items)
        rng = rng_for(view.prompt, "order")
        true = sorted(items, key=lambda s: pos[s.iid])
        if self.real:
            keep = 0.5 if n <= 8 else 0.35 if n <= 16 else 0.25
            runs, cur = [], [true[0]]
            for prev, x in zip(true, true[1:], strict=False):
                if pos[x.iid] == pos[prev.iid] + 1 and rng.random() < keep:
                    cur.append(x)
                else:
                    runs.append(cur)
                    cur = [x]
            runs.append(cur)
            rng.shuffle(runs)
            res = [x for run in runs for x in run]
            n_drop = 0 if rng.random() > (0.15 if n <= 8 else 0.3) else rng.randint(1, 2) if n <= 16 else rng.randint(6, 10)
            for _ in range(min(n_drop, len(res) - 1)):
                res.pop(rng.randrange(len(res)))
            return json.dumps([label_value(s.label) for s in res])
        if n <= 6:
            res = list(true)
            if rng.random() > 0.85 and n >= 2:
                i = rng.randrange(n - 1)
                res[i], res[i + 1] = res[i + 1], res[i]
        elif n <= 16:
            res = list(true)
            for _ in range(max(1, n // 6)):
                i = rng.randrange(n - 1)
                res[i], res[i + 1] = res[i + 1], res[i]
        else:
            run = rng.randint(2, 4)
            chunks = [true[i:i + run] for i in range(0, n, run)]
            rng.shuffle(chunks)
            res = [s for c in chunks for s in c]
            if n > 40:
                res = [s for s in res if rng.random() > 0.08]
        return json.dumps([label_value(s.label) for s in res])

    def _order_pieces(self, view: View, f: dict, pieces: list, pos: dict, st: dict) -> str:
        n = len(pieces)
        rng = rng_for(view.prompt, "pieces")
        key = [(min(pos[s.iid] for s in members), lab) for lab, members in pieces]
        true = [lab for _, lab in sorted(key)]
        if n == 2:
            gap = abs(key[0][0] - key[1][0])
            if self.real:
                ends = sorted((min(pos[s.iid] for s in m), max(pos[s.iid] for s in m)) for _, m in pieces)
                between = ends[1][0] - ends[0][1] - 1
                p_ok = 0.9 if between <= 0 else 0.67 if between <= 3 else 0.5
            else:
                p_ok = (0.94 if gap <= 3 else 0.98) - self.err_prob(view.prompt)
            ok = rng.random() < p_ok
            first = true[0] if ok else true[1]
            keys = _pick_json_keys(view)
            if keys:
                return json.dumps(fill_keys(keys, {"id": label_value(first), "answer": label_value(first),
                                                   "verdict": label_value(first), "text": label_value(first)}))
            if not has_any(view.low, r"\border\b", r"\blist\b", r"\bsequence\b"):
                return f"Piece {first} comes first."
        res = list(true)
        if self.real:
            spans = {lab: (min(pos[s.iid] for s in m), max(pos[s.iid] for s in m)) for lab, m in pieces}
            runs, cur = [], [true[0]]
            for a, b in zip(true, true[1:], strict=False):
                if spans[b][0] == spans[a][1] + 1 and rng.random() < 0.35:
                    cur.append(b)
                else:
                    runs.append(cur)
                    cur = [b]
            runs.append(cur)
            rng.shuffle(runs)
            return json.dumps([label_value(x) for run in runs for x in run])
        p_swap = 0.04 if n <= 8 else 0.08 if n <= 20 else 0.15
        for i in range(n - 1):
            if rng.random() < p_swap:
                res[i], res[i + 1] = res[i + 1], res[i]
        if n > 30:
            st["agg_requests"] += 1
            run = rng.randint(3, 6)
            chunks = [res[i:i + run] for i in range(0, n, run)]
            rng.shuffle(chunks)
            res = [x for c in chunks for x in c]
        return json.dumps([label_value(x) for x in res]) if wants_json(view.instr) or True else ""


def _augment_partials(view: View, texts: dict, k: int = 48) -> None:
    """Find passages shown only by their first / last ``k`` characters and add them as partial occurrences."""
    from .oracle_base import Seen, find_label
    prompt = view.prompt
    spans = [(s.pos, s.end) for s in view.items]
    def covered(p: int) -> bool:
        return any(a - 400 <= p <= b for a, b in spans if b - a > 400) or any(a - 20 <= p <= b for a, b in spans)
    added = []
    for iid, t in texts.items():
        t = t.strip()
        if len(t) < 2 * k:
            continue
        for frag, is_head in ((t[:k], True), (t[-k:], False)):
            start = 0
            while True:
                p = prompt.find(frag, start)
                if p < 0:
                    break
                start = p + 1
                if covered(p):
                    continue
                q = p if is_head else max(0, p - 300)
                hdr = prompt[max(0, q - 120):q] if is_head else prompt[max(0, q - 120):p]
                lab = find_label(prompt[max(0, p - 120):p]) if is_head else find_label(hdr)
                added.append(Seen(iid=iid, label=lab or "", pos=p, end=p + k, cov=0.2, full=False))
                spans.append((p, p + k))
    if added:
        view.items = sorted(view.items + added, key=lambda x: x.pos)
        for x in view.items:
            view.by_iid.setdefault(x.iid, x)
            if x.label:
                view.by_label.setdefault(x.label, x)


_PIECE_HDR = re.compile(
    r"(?im)^[ \t>*#\-]*(?:\*\*)?(?:(?P<kw>piece|chunk|segment|block|chain|group|cluster|part|set|run)\s*#?\s*)?"
    r"(?P<lab>[A-Z]{0,2}\d{1,5})\b(?:\*\*)?\s*[:.)\-]?")


def _pieces_of(view: View) -> list[tuple[str, list[Seen]]]:
    hdrs = [(m.start(), m.group("lab")) for m in _PIECE_HDR.finditer(view.prompt)
            if m.group("kw") or re.match(r"[A-Za-z]", m.group("lab"))]
    if len(hdrs) < 2:
        return []
    groups: dict[str, list[Seen]] = defaultdict(list)
    order = []
    for s in view.items:
        lab = None
        for p, h in hdrs:
            if p <= s.pos:
                lab = h
            else:
                break
        if lab is None:
            continue
        if lab not in groups:
            order.append(lab)
        groups[lab].append(s)
    if all(len(groups[k]) == 1 and groups[k][0].label == k for k in order):
        return []
    return [(k, groups[k]) for k in order]


def _clean(h: str) -> str:
    m = re.match(r"(?i)(?:piece|chunk|segment|block|chain|group|part)\s*#?\s*(\d+)", h.strip())
    return m.group(1) if m else h.strip().replace(" ", "")


POS_WORDS: dict[str, frozenset] = {}


def _lexicon() -> dict[str, frozenset]:
    if not POS_WORDS:
        try:
            import importlib
            g = importlib.import_module("ctcbench_real._vendor.ctc.tasks.textgroups.generate")
            POS_WORDS.update(nouns=frozenset(g.NOUNS), verbs=frozenset(g.VERBS), adjectives=frozenset(g.ADJECTIVES),
                             adverbs=frozenset(g.ADVERBS))
        except Exception:  # noqa: BLE001  fall back to the lexicon below
            POS_WORDS.update(_FALLBACK_LEX)
    return POS_WORDS


_FALLBACK_LEX = {
    "nouns": frozenset("""river mountain teacher engine harbor garden lantern soldier merchant valley bridge castle
farmer anchor orchard painter compass meadow captain tavern forest blacksmith fountain carriage shepherd windmill
courtyard violin telescope harvest glacier cathedral scholar musician voyage saddle beacon quarry chapel wagon sailor
tower island desert canyon miner weaver potter hunter baker library market palace cottage harvester watchman gardener
fisherman innkeeper drummer""".split()),
    "verbs": frozenset("""wandered shouted gathered vanished trembled lingered stumbled whispered marched collapsed
drifted arrived departed shattered glimmered echoed faltered soared crumbled scattered returned circled hesitated rushed
paused rested waited knelt listened watched rebuilt explored guarded studied celebrated argued labored traveled
sketched harvested stood""".split()),
    "adjectives": frozenset("""ancient weary golden silent restless fragile stormy distant crimson hollow gentle ragged
luminous narrow frozen endless bitter radiant crooked solemn vivid humble fierce tranquil scarlet rugged mournful brisk
lavish somber""".split()),
    "adverbs": frozenset("quietly suddenly slowly wearily eagerly calmly abruptly softly bravely restlessly".split()),
}


def _feature_of(low: str, default: str) -> str:
    for k, pat in (("adjectives", r"adjective"), ("nouns", r"\bnouns?\b"), ("verbs", r"\bverbs?\b"),
                   ("adverbs", r"adverb")):
        if re.search(pat, low):
            return k
    return default


def count_text(text: str, feature: str, word: str | None) -> int:
    toks = [t.strip(".,;:!?\"'()").lower() for t in text.split()]
    if feature == "connector":
        return sum(1 for t in toks if t == (word or "").lower())
    lex = _lexicon().get(feature) or frozenset()
    return sum(1 for t in toks if t in lex)


class TextgroupsOracle(CTCOracle):
    """textgroups: passages of template sentences; values are feature counts, groups of 3 sum to the target."""

    def respond(self, prompt: str, hidden: dict, st: dict) -> str:
        from .oracle_base import note_call
        note_call(st, prompt, len(self._passages_in(prompt, hidden)), len(hidden.get("items") or []))
        body = self.answer_text(prompt, hidden, st)
        return self.chat(prompt, body)

    def _passages_in(self, prompt: str, hidden: dict) -> list[tuple[str, str, int, int]]:
        """(passage id, label, start, end) of every passage shown in full, in prompt order."""
        out = []
        for iid, text in hidden["items"]:
            head = text[:80]
            p = prompt.find(head)
            if p < 0:
                continue
            tail = text[-60:]
            q = prompt.find(tail, p)
            if q < 0:
                continue
            hdr = prompt[max(0, p - 60):p]
            from .oracle_base import find_label
            out.append((str(iid), find_label(hdr) or "", p, q + len(tail)))
        out.sort(key=lambda x: x[2])
        res = []
        for k, (iid, lab, a, b) in enumerate(out):
            res.append((iid, lab or str(k + 1), a, b))
        return res

    def answer_text(self, prompt: str, hidden: dict, st: dict) -> str:
        f = hidden
        low = prompt.lower()
        feat = f["feature"] if f["feature"] != "connector" else "connector"
        asked = _feature_of(low, feat)
        shown = self._passages_in(prompt, hidden)
        st["calls_tg"] = st.get("calls_tg", 0) + 1
        if shown:
            reads = st.setdefault("read_ids", {})
            for iid, *_ in shown:
                reads[iid] = reads.get(iid, 0) + 1
            st["items_read"] = len(reads)
        if has_any(low, r"(sum|add up|total).{0,40}\b(70|target)\b", r"groups? of (3|three)", r"triples?"):
            st["agg_requests"] += 1
            rng = rng_for(prompt, "tg-agg")
            ids = [lab for _, lab, _, _ in shown] or ["1", "2", "3"]
            k = min(3, len(ids))
            return json.dumps([[label_value(x) for x in rng.sample(ids, k)]]) if len(ids) >= 3 else "[]"
        if shown:
            bump(st, "passage_count_calls")
            n = len(shown)
            out = []
            for iid, lab, a, b in shown:
                true = count_text(prompt[a:b], asked if asked != "connector" else "connector", f.get("word"))
                if asked != feat and feat != "connector":
                    true = count_text(prompt[a:b], asked, None)
                r = rng_for(prompt, iid, "cnt")
                sd = 0.12 * true + 1.5 + 0.02 * n
                val = max(0, int(round(true + r.gauss(0, sd))))
                out.append((lab, val))
            return render_per_item(out, prompt)
        words = re.findall(r"\b[a-z]{3,}\b", prompt)
        lex = _lexicon()
        known = [w for w in dict.fromkeys(words) if any(w in v for v in lex.values())]
        if len(known) >= 4:
            bump(st, "word_class_calls")
        if len(known) >= 4 and has_any(low, r"which (of these )?words", r"classify", r"part of speech", r"\bpos\b",
                                       r"label each word", r"for each word", r"is each"):
            if has_any(low, r"for each word", r"label each", r"part of speech", r"classify each", r"\btag\b"):
                tags = []
                for w in known:
                    tag = next((k[:-1] for k, v in lex.items() if w in v), "other")
                    if rng_for(prompt, w, "pos").random() < 0.01:
                        tag = "noun" if tag != "noun" else "adjective"
                    tags.append((w, tag))
                return render_per_item(tags, prompt)
            sel = [w for w in known if w in lex.get(asked, frozenset())]
            return json.dumps(sel) if wants_json(prompt) else ", ".join(sel)
        body = prompt
        if has_any(low, r"how many|count|number of"):
            true = count_text(body, asked if asked != "connector" else "connector", f.get("word"))
            r = rng_for(prompt, "short")
            n_words = len(body.split())
            val = true if r.random() < (0.95 if n_words < 80 else 0.7) else true + r.choice([-1, 1])
            return str(max(0, val))
        if has_any(low, r"list (all|the|every)"):
            lexw = lex.get(asked, frozenset())
            found = [t.strip(".,").lower() for t in body.split() if t.strip(".,").lower() in lexw]
            return json.dumps(found) if wants_json(prompt) else ", ".join(found)
        return refuse_aggregate(st)


ORACLES = {
    "nq": RetrievalOracle, "hotpotqa": RetrievalOracle, "qdmatch_hpqa": QDMatchOracle,
    "contradiction_iid": ContradictionOracle, "outlier": OutlierOracle, "outlier_fixedM": OutlierOracle,
    "grouping": GroupingOracle, "reorder": ReorderOracle, "textgroups": TextgroupsOracle,
}

READ_MODELS = {
    "nq": ReadModel(budget_chars=400_000, reliable_items=40, miss_rate=0.5, base_err=0.02, load_err=0.25,
                    agg_capacity=10, chatty=0.15),
    "hotpotqa": ReadModel(budget_chars=400_000, reliable_items=40, miss_rate=0.5, base_err=0.02, load_err=0.25,
                          agg_capacity=10, chatty=0.15),
    "qdmatch_hpqa": ReadModel(budget_chars=400_000, reliable_items=40, miss_rate=0.5, base_err=0.02, load_err=0.25,
                              agg_capacity=10, chatty=0.1),
    "contradiction_iid": ReadModel(budget_chars=400_000, reliable_items=60, miss_rate=0.4, base_err=0.01,
                                   load_err=0.3, agg_capacity=10, chatty=0.1),
    "outlier": ReadModel(budget_chars=400_000, reliable_items=50, miss_rate=0.5, base_err=0.02, load_err=0.25,
                         agg_capacity=10, chatty=0.15),
    "outlier_fixedM": ReadModel(budget_chars=400_000, reliable_items=50, miss_rate=0.5, base_err=0.02,
                                load_err=0.25, agg_capacity=10, chatty=0.15),
    "grouping": ReadModel(budget_chars=400_000, reliable_items=40, miss_rate=0.5, base_err=0.02, load_err=0.25,
                          agg_capacity=10, chatty=0.15),
    "reorder": ReadModel(budget_chars=400_000, reliable_items=40, miss_rate=0.4, base_err=0.01, load_err=0.2,
                         agg_capacity=8, chatty=0.1),
    "textgroups": ReadModel(budget_chars=400_000, reliable_items=40, miss_rate=0.4, base_err=0.01, load_err=0.2,
                            agg_capacity=8, chatty=0.1),
}


def oracle_for(family: str, reorder_oracle: str = "real", grouping_oracle: str = "real",
               contradiction_oracle: str = "real") -> CTCOracle:
    if family == "reorder":
        return ReorderOracle(READ_MODELS[family], strength=reorder_oracle)
    if family == "contradiction_iid":
        return ContradictionOracle(READ_MODELS[family], strength=contradiction_oracle)
    if family == "grouping" and grouping_oracle == "real":
        from .grouping_real import (
            GroupingOracleReal,
        )
        return GroupingOracleReal(READ_MODELS[family])
    return ORACLES[family](READ_MODELS[family])


__all__ = ["ORACLES", "READ_MODELS", "count_text", "label_variants", "oracle_for"]
