"""Grouping oracle calibrated to the real 27B sub-LM (thinking off)."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, OrderedDict

import numpy as np
from synth_common.oracle import refuse_aggregate, rng_for, wants_json

from .oracle_base import View, bump, label_value, render_per_item, texts_of
from .oracles import GroupingOracle, canon_of, has_any

LEVEL = {1: dict(ratio=2.9, same_yes=.55, diff_no=.94, file_err=.37),
         2: dict(ratio=2.1, same_yes=.38, diff_no=.94, file_err=.27),
         3: dict(ratio=1.6, same_yes=.20, diff_no=1.0, file_err=.31)}
DISCIPLINE_RATIO = .55
SUBFIELD_RATIO = 1.7
DRIFT = {"free": .34, "discipline": .19, "subfield": .5}
MERGE_SPREAD = .35
FILE_MARGIN = .05

_TOK = re.compile(r"[a-z][a-z0-9\-]{2,}")
_STOP = set("the and for with that this from are was were have has been into their its our which these those using "
            "used based study studies results result paper also than between both such can may more most other new "
            "two three one however here show shows shown well within among".split())
_CACHE: OrderedDict = OrderedDict()


def _tokens(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


class _Lex:
    """tf-idf vectors of a row's papers (title counted twice), and partitions of them by average-linkage clustering."""

    def __init__(self, texts: dict[str, str]):
        self.ids = sorted(texts, key=lambda x: int(x) if x.isdigit() else x)
        docs = []
        for i in self.ids:
            title, _, body = texts[i].partition("\n")
            docs.append(_tokens(title) * 2 + _tokens(body))
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.vocab = sorted(t for t, c in df.items() if c < 0.5 * n)
        self.col = {t: j for j, t in enumerate(self.vocab)}
        self.idf = np.array([math.log(n / df[t]) for t in self.vocab])
        m = np.zeros((n, len(self.vocab)))
        for r, d in enumerate(docs):
            for t, c in Counter(d).items():
                j = self.col.get(t)
                if j is not None:
                    m[r, j] = (1 + math.log(c)) * self.idf[j]
        self.m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
        self.parts: dict[int, dict] = {}
        self.named: dict[str, set[str]] = {}

    def register(self, name: str, iids) -> None:
        self.named.setdefault(name.strip().lower(), set()).update(iids)

    def option_vec(self, name: str) -> np.ndarray:
        """What a category name means to the sub-LM: the papers behind a name it produced, else the name's words."""
        rows = [self.ids.index(i) for i in self.named.get(name.strip().lower(), ()) if i in self.ids]
        if rows:
            v = self.m[rows].mean(axis=0)
            return v / max(np.linalg.norm(v), 1e-9)
        return self.vec_of(name)

    def vec_of(self, text: str) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t, c in Counter(_tokens(text)).items():
            j = self.col.get(t)
            if j is not None:
                v[j] = (1 + math.log(c)) * self.idf[j]
        return v / max(np.linalg.norm(v), 1e-9)

    def name_of(self, rows: list[int]) -> str:
        cen = self.m[rows].sum(axis=0)
        top = [self.vocab[j] for j in np.argsort(-cen)[:3] if cen[j] > 0]
        return " ".join(w.capitalize() for w in top) or "General"

    def partition(self, k: int) -> dict:
        k = max(1, min(k, len(self.ids)))
        if k not in self.parts:
            groups = _agglo(self.m, k)
            of, names, members = {}, {}, {}
            for c, rows in enumerate(groups):
                names[c] = self.name_of(rows)
                members[c] = [self.ids[r] for r in rows]
                self.register(names[c], members[c])
                for r in rows:
                    of[self.ids[r]] = c
            cen = np.stack([self.m[rows].mean(axis=0) for rows in groups])
            sims = self.m @ cen.T
            alt = {}
            for r, i in enumerate(self.ids):
                sims[r, of[i]] = -np.inf
                alt[i] = int(np.argmax(sims[r])) if len(groups) > 1 else of[i]
            self.parts[k] = {"of": of, "alt": alt, "name": names, "members": members}
        return self.parts[k]

    def label(self, k: int, iid: str, kind: str) -> str:
        """The paper's label in the k-cluster partition, drifted to its nearest other cluster w.p. DRIFT[kind]."""
        part = self.partition(k)
        c = part["alt"][iid] if rng_for("lexdrift", kind, k, iid).random() < DRIFT[kind] else part["of"][iid]
        return part["name"][c]


def _agglo(m: np.ndarray, k: int) -> list[list[int]]:
    """Average-linkage agglomerative clustering on cosine similarity, down to k clusters."""
    n = m.shape[0]
    sim = m @ m.T
    np.fill_diagonal(sim, -np.inf)
    size = np.ones(n)
    alive = np.ones(n, dtype=bool)
    members = [[i] for i in range(n)]
    for _ in range(n - k):
        s = np.where(alive[:, None] & alive[None, :], sim, -np.inf)
        a, b = divmod(int(np.argmax(s)), n)
        if a > b:
            a, b = b, a
        new = (sim[a] * size[a] + sim[b] * size[b]) / (size[a] + size[b])
        sim[a, :], sim[:, a] = new, new
        sim[a, a] = -np.inf
        sim[b, :], sim[:, b] = -np.inf, -np.inf
        size[a] += size[b]
        alive[b] = False
        members[a] += members[b]
        members[b] = []
    return [g for g in members if g]


def lex_for(hidden: dict) -> _Lex:
    texts = texts_of(hidden)
    key = (len(texts), tuple(sorted(texts)[:3]), tuple(texts[i][:40] for i in sorted(texts)[:3]))
    hit = _CACHE.get(key)
    if hit is None:
        hit = _Lex(texts)
        _CACHE[key] = hit
        while len(_CACHE) > 64:
            _CACHE.popitem(last=False)
    _CACHE.move_to_end(key)
    return hit


class GroupingOracleReal(GroupingOracle):
    """The grouping sub-LM, at the real model's measured error rates."""

    def answer(self, view: View, hidden: dict, st: dict) -> str:
        f = hidden
        low = view.low
        lvl = min(max(int(f.get("level") or 1), 1), 3)
        P = LEVEL[lvl]
        K = int(f["K"])
        lex = lex_for(f)
        if not view.items:
            if has_any(low, r"merge", r"synonym", r"canonical", r"map (each|these)", r"consolidat", r"normali"):
                return self._merge_real(view, lex, st, K)
            return refuse_aggregate(st)
        read, _ = self.read_view(view, st)
        lines = list(re.finditer(r"(?m)^\s*(?:Cluster|Group|Set)\s+(\w+)\s*:", view.prompt))
        if len(lines) >= 2 and has_any(low, r"\bname\b", r"\btopic\b", r"\btitle\b"):
            bump(st, "name_calls")
            out = {}
            for k, m in enumerate(lines):
                end = lines[k + 1].start() if k + 1 < len(lines) else len(view.prompt)
                mem = [x for x in view.items if m.end() <= x.pos < end and x.iid in lex.ids]
                name = lex.name_of([lex.ids.index(x.iid) for x in mem]) if mem else "General"
                lex.register(name, [x.iid for x in mem])
                out[m.group(1)] = name
            return json.dumps(out, ensure_ascii=False) if wants_json(view.instr) or "{" in view.instr else \
                "\n".join(f"Cluster {k}: {v}" for k, v in out.items())
        if len(read) == 2 and has_any(low, r"same (topic|field|category|group|cluster|research|area|subject|concept|"
                                           r"subfield|discipline)", r"belong (to|in) the same", r"should .* be grouped"):
            a, b = read
            truth = f["label"][a.iid] == f["label"][b.iid]
            r = rng_for(view.prompt, "same").random()
            return "YES" if ((r < P["same_yes"]) if truth else (r >= P["diff_no"])) else "NO"
        if has_any(low, r"\bname\b.{0,40}\b(group|cluster|set|these papers|them)\b", r"(shared|common) (research )?topic",
                   r"what (topic|field|subject) do these", r"one (short )?(name|title|label) for (all|these|the group)"):
            bump(st, "name_calls")
            name = lex.name_of([lex.ids.index(s.iid) for s in read if s.iid in lex.ids]) if read else "General"
            lex.register(name, [s.iid for s in read])
            return json.dumps({"name": name}) if wants_json(view.instr) else name
        if has_any(low, r"\b(cluster|partition|group)\b.*\b(these|the following|them)\b") and \
                not has_any(low, r"\blabel\b", r"for each (paper|abstract|document)"):
            st["agg_requests"] += 1
            part = lex.partition(round(K * len(read) / max(1, len(lex.ids))) or 1)
            groups: dict[int, list] = {}
            for s in read:
                groups.setdefault(part["of"][s.iid], []).append(label_value(s.label))
            if wants_json(view.instr):
                return json.dumps({"groups": [{"label": part["name"][c], "doc_ids": v} for c, v in groups.items()]})
            return "\n".join(f"{part['name'][c]}: {v}" for c, v in groups.items())
        opts = self._all_options(view) or self._options(view, set(f["label"].values()))
        if len(opts) >= 2:
            bump(st, "fixed_list_calls")
            return self._file_real(view, f, lex, read, opts, P)
        bump(st, "free_label_calls")
        two = has_any(low, r"two[- ]level", r"discipline.*subfield", r"field.*subfield", r"broad.*specific")
        kf = round((SUBFIELD_RATIO if two else P["ratio"]) * K)
        kc = round(DISCIPLINE_RATIO * K)
        out = []
        for s in read:
            v = lex.label(kf, s.iid, "subfield" if two else "free")
            out.append((s.label, [lex.label(kc, s.iid, "discipline"), v] if two else v))
        coarse = two
        if coarse and re.search(r"\[\[\s*<?\s*n", view.instr):
            return json.dumps([[label_value(lab), *v] for lab, v in out], ensure_ascii=False)
        return render_per_item(out, view.instr)

    @staticmethod
    def _all_options(view: View) -> list[str]:
        """Every bulleted / numbered line of the instruction, so long category lists are read in full."""
        opts = []
        for m in re.finditer(r"(?m)^\s*(?:\d{1,3}[.):]|[-*•])\s*(.+?)\s*$", view.instr):
            o = m.group(1).strip().strip('"')
            if 2 <= len(o) <= 120 and len(o.split()) <= 12 and not re.search(r"(?i)\b(reply|json|answer|return)\b", o):
                opts.append(o)
        return list(dict.fromkeys(opts))

    def _file_real(self, view: View, f: dict, lex: _Lex, read: list, opts: list[str], P: dict) -> str:
        labels = set(f["label"].values())
        canon = {o: canon_of(o, labels) for o in opts}
        ovec = np.stack([lex.option_vec(re.sub(r"\s*\(\d+\)\s*$", "", o)) for o in opts])
        num = re.search(r"(?:reply|answer|give|return)[^.\n]{0,40}\b(number|index)\b", view.low)
        out = []
        for s in read:
            gold = f["label"][s.iid]
            hit = [o for o in opts if canon[o] == gold]
            sims = ovec @ lex.m[lex.ids.index(s.iid)] if s.iid in lex.ids else np.zeros(len(opts))
            idx = np.argsort(-sims)
            order = [opts[j] for j in idx]
            right = hit[0] if hit else order[0]
            margin = float(sims[idx[0]] - sims[idx[1]]) if len(idx) > 1 else 1.0
            err = min(0.95, P["file_err"] * 2.0 * math.exp(-margin / FILE_MARGIN))
            if rng_for(view.prompt, s.iid, "file").random() >= err:
                val = right
            else:
                alt = [o for o in order if o != right] or opts
                val = alt[0] if rng_for(view.prompt, s.iid, "alt").random() < .6 else \
                    alt[rng_for(view.prompt, s.iid, "pick").randrange(len(alt))]
            out.append((s.label, (opts.index(val) + 1) if num else val))
        return render_per_item(out, view.instr)

    def _merge_real(self, view: View, lex: _Lex, st: dict, K: int) -> str:
        bump(st, "merge_calls")
        cand = re.findall(r'"([^"\n]{2,80})"', view.instr) or \
            [x.strip(" -*•\t") for x in view.instr.split("\n") if 2 <= len(x.strip(" -*•\t")) <= 80]
        cand = list(dict.fromkeys(re.sub(r"\s*\(\d+\)\s*$|\s*[:\-]\s*\d+\s*$", "", c).strip() for c in cand))[:200]
        if not cand:
            return refuse_aggregate(st)
        m = re.search(r"exactly (\d+)|(\d+) (?:groups|categories|clusters)", view.low)
        asked = int(m.group(1) or m.group(2)) if m else K
        r = rng_for(view.prompt, "merge")
        target = max(1, min(len(cand), round(asked * math.exp(r.gauss(0, MERGE_SPREAD)))))
        vec = np.stack([lex.option_vec(c) for c in cand])
        groups = _agglo(vec, target) if len(cand) > target else [[i] for i in range(len(cand))]
        mapping = {}
        for g in groups:
            head = cand[g[0]]
            for i in g:
                mapping[cand[i]] = head
                lex.register(head, lex.named.get(cand[i].strip().lower(), ()))
        return json.dumps(mapping, ensure_ascii=False) if wants_json(view.instr) else \
            "\n".join(f"{k} -> {v}" for k, v in mapping.items())


__all__ = ["DISCIPLINE_RATIO", "LEVEL", "GroupingOracleReal", "lex_for"]
