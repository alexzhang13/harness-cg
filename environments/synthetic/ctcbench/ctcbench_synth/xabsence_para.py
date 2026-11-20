"""Synthetic ``xabsence_paraphrase`` family: cross-corpus absence over PubMed paraphrases."""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import re
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

from synth_common.oracle import FakeSubLM, ReadModel, rng_for
from rlm_train.frozen import frozen_path

SIDECAR = Path(os.environ.get("XABSENCE_PARA_POOL") or Path(__file__).with_name("sidecars") / "xabsence_para_pool.json.gz")
FAMILY = "xabsence_paraphrase"
RUNG_PAIRS = {"native": 333, "32k": 333, "16k": 165, "8k": 81, "4k": 39}
SIDE_MIX = (("ABB", 0.392), ("AAB", 0.344), ("BBB", 0.144), ("AAA", 0.12))

P_YES_TRUE = {"mutual": 0.94, "nonmutual": 0.78}
P_YES_LOOKALIKE, P_YES_ORPHAN, P_YES_UNRELATED = 0.015, 0.033, 0.003
FIND = {"hard": (0.97, 11000.0), "mid": (0.995, 1e9), "easy": (0.985, 1e9)}
WRONG = {"listed": 0.5, "absent_10": 0.003, "absent_long": 0.083}
P_UNFINDABLE = 0.0
PERSIST = {"judge": 0.87, "pick": 0.5}

_STOP = set("the a an of and or in on to for with by is are was were be been that this these those as at from it its "
            "which we our their than not no".split())


def _stem(w: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def _toks(s: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP]


def _norm(s: str) -> str:
    return " ".join(str(s).strip().strip('"').strip("'").lower().split())


@cache
def pool(split: str) -> tuple[dict, ...]:
    with gzip.open(frozen_path(SIDECAR, "ctcbench/xabsence_para_pool.json.gz"), "rt", encoding="utf-8") as f:
        d = json.load(f)
    return tuple(a for a in d["abstracts"] if a["split"] == split)


@cache
def _decoy_index(split: str) -> tuple[list, dict]:
    """(pairs, token -> pair indices) over the split's pool, for orphan decoys."""
    pairs = [(a["id"], o, p) for a in pool(split) for o, p in a["pairs"]]
    inv: dict[str, list[int]] = {}
    for k, (_aid, o, _p) in enumerate(pairs):
        for w in set(_toks(o)):
            inv.setdefault(w, []).append(k)
    return pairs, inv


def _decoys(split: str, text: str, aid: str, n: int, taken: set[str], rng: random.Random) -> list[tuple]:
    """``n`` look-alike pairs from other abstracts (like the generator's ``decoys_per_unmatched``)."""
    pairs, inv = _decoy_index(split)
    df = {w: len(inv.get(w, ())) for w in set(_toks(text))}
    score: Counter = Counter()
    for w, d in df.items():
        if 0 < d <= 2000:
            for k in inv[w]:
                score[k] += 1.0 / math.log(2 + d)
    out = []
    for k, _ in score.most_common(50):
        a, o, p = pairs[k]
        if a != aid and _norm(o) not in taken and _norm(p) not in taken:
            out.append(pairs[k])
            taken.update((_norm(o), _norm(p)))
            if len(out) == n:
                break
    return out


def compose(split: str, row: int, rung: str = "native", seed: int = 0, decoys: int = 2,
            per_abstract: int = 3) -> dict:
    """One unified-format xabsence example (documents A block then B block, 0-based gold)."""
    n_pairs = RUNG_PAIRS[rung]
    rng = random.Random(f"xabpara:{split}:{row}:{rung}:{seed}")
    abstracts = list(pool(split))
    rng.shuffle(abstracts)
    n_base = n_pairs + 3 - 3 * decoys
    sents: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for a in abstracts:
        prs = list(a["pairs"])
        if per_abstract:
            prs = rng.sample(prs, min(per_abstract, len(prs)))
        for o, p in prs:
            if _norm(o) in seen or _norm(p) in seen:
                continue
            seen.update((_norm(o), _norm(p)))
            sents.append((a["id"], o, p))
        if len(sents) >= n_base:
            break
    if len(sents) < n_base and per_abstract:
        for a in abstracts:
            for o, p in a["pairs"]:
                if len(sents) >= n_base:
                    break
                if _norm(o) not in seen and _norm(p) not in seen:
                    seen.update((_norm(o), _norm(p)))
                    sents.append((a["id"], o, p))
    sents = sents[:n_base]
    pattern = rng.choices([s for s, _ in SIDE_MIX], [w for _, w in SIDE_MIX])[0]
    orph = rng.sample(range(len(sents)), 3)
    for k in orph:
        side = pattern[orph.index(k)]
        aid, o, p = sents[k]
        sents += _decoys(split, o if side == "A" else p, aid, decoys, seen, rng)
    A, B = [], []
    for k, (_aid, o, p) in enumerate(sents):
        if k in orph:
            side = pattern[orph.index(k)]
            (A if side == "A" else B).append((o if side == "A" else p, True, k))
        else:
            A.append((o, False, k))
            B.append((p, False, k))
    rng.shuffle(A)
    rng.shuffle(B)
    docs = [{"corpus": "A", "text": t} for t, _, _ in A] + [{"corpus": "B", "text": t} for t, _, _ in B]
    gold = [i for i, (_, o, _) in enumerate(A + B) if o]
    pair_of = [k for _, _, k in A + B]
    return {"documents": docs, "queries": [], "gold_doc_indices": gold, "num_pairs": len(sents) - 3,
            "num_unmatched": 3, "source": "xabsence_synth_pubmed",
            "_meta": {"pair_of": pair_of, "pattern": pattern, "decoys_per_unmatched": decoys}}


def _tfidf(texts: dict[int, str]) -> dict[int, dict[str, float]]:
    T = {i: Counter(_toks(t)) for i, t in texts.items()}
    df = Counter(w for c in T.values() for w in c)
    n = len(T)
    out = {}
    for i, c in T.items():
        v = {w: (1 + math.log(k)) * math.log((n + 1) / (df[w] + 1) + 1) for w, k in c.items()}
        z = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out[i] = {w: x / z for w, x in v.items()}
    return out


def facts(ex: dict) -> dict:
    """Oracle facts keyed by 1-based claim id: partner (or None), side, partner cosine bucket, text map."""
    docs = ex["documents"]
    pair_of = ex["_meta"]["pair_of"]
    by_pair: dict[int, list[int]] = {}
    for i, k in enumerate(pair_of):
        by_pair.setdefault(k, []).append(i + 1)
    partner = {}
    for ids in by_pair.values():
        if len(ids) == 2:
            partner[ids[0]], partner[ids[1]] = ids[1], ids[0]
    texts = {i + 1: d["text"] for i, d in enumerate(docs)}
    V = _tfidf(texts)
    inv: dict[str, list] = {}
    for j, v in V.items():
        for w, x in v.items():
            inv.setdefault(w, []).append((j, x))
    side = {i + 1: d["corpus"] for i, d in enumerate(docs)}
    nn = {}
    for i, v in V.items():
        acc: Counter = Counter()
        for w, x in v.items():
            for j, y in inv[w]:
                if side[j] != side[i]:
                    acc[j] += x * y
        nn[i] = max(acc, key=lambda j: (acc[j], -j)) if acc else None
    bucket, mutual = {}, []
    for i, j in partner.items():
        c = sum(x * V[j].get(w, 0.0) for w, x in V[i].items())
        bucket[str(i)] = "hard" if c < 0.15 else "mid" if c < 0.35 else "easy"
        if nn.get(i) == j and nn.get(j) == i:
            mutual.append(str(i))
    return {"partner": {str(i): j for i, j in partner.items()}, "side": {str(i + 1): d["corpus"] for i, d in enumerate(docs)},
            "bucket": bucket, "mutual": mutual, "orphans": [g + 1 for g in ex["gold_doc_indices"]],
            "tmap": {_norm(t): i for i, t in texts.items()}}


_TEXT_CACHE: dict[int, tuple[dict, dict]] = {}


def _texts(tmap: dict) -> dict:
    hit = _TEXT_CACHE.get(id(tmap))
    if hit is None or hit[0] is not tmap:
        if len(_TEXT_CACHE) > 256:
            _TEXT_CACHE.clear()
        hit = (tmap, {i: t for t, i in tmap.items()})
        _TEXT_CACHE[id(tmap)] = hit
    return hit[1]

_SPLITS = re.compile(r"(?:[:\]\)\.>]\s+|\t|\s-\s)")
_LABEL = re.compile(r"\[(\d{1,5})\]")
_PAIR_A = re.compile(r"(?i)^\s*(?:pair\s*)?#?(\d{1,4})\s*[.:)\]]\s*(?:\(?a\)?|claim a|corpus a)\s*[:)\]]")
_PAIR_B = re.compile(r"(?i)^\s*(?:\(?b\)?|claim b|corpus b)\s*[:)\]]")


def _match_line(s: str, tmap: dict) -> tuple[int | None, str]:
    """(claim id, the line's prefix before the claim text) for a line that ends with a claim."""
    iid = tmap.get(_norm(s))
    if iid is not None:
        return iid, ""
    for m in _SPLITS.finditer(s):
        iid = tmap.get(_norm(s[m.end():]))
        if iid is not None:
            return iid, s[:m.end()]
    return None, ""


def _wants_json(prompt: str) -> bool:
    return "json" in prompt.lower() or "{" in prompt


class XAbsenceParaOracle(FakeSubLM):
    def __init__(self) -> None:
        super().__init__(ReadModel(budget_chars=400_000, reliable_items=1000, chatty=0.1))

    def _u(self, key: Any, prompt: str) -> float:
        if rng_for("xabp-persist", key, prompt).random() < PERSIST["judge" if key[0] == "j" else "pick"]:
            return rng_for("xabp", key).random()
        return rng_for("xabp", key, prompt).random()

    def judge(self, a: int, b: int, F: dict, prompt: str) -> bool:
        pa = F["partner"].get(str(a))
        orph = set(F["orphans"])
        if pa == b:
            p = 0.0 if self._unfindable(a, F) else P_YES_TRUE["mutual" if str(a) in F["mutual"] else "nonmutual"]
        elif a in orph or b in orph:
            p = P_YES_ORPHAN
        else:
            p = P_YES_LOOKALIKE if self._lookalike(a, b, F) else P_YES_UNRELATED
        return self._u(("j", min(a, b), max(a, b)), prompt) < p

    def _unfindable(self, a: int, F: dict) -> bool:
        p = F["partner"].get(str(a))
        return p is not None and rng_for("xabp-unfindable", min(a, p), max(a, p), F["orphans"]).random() < P_UNFINDABLE

    def _lookalike(self, a: int, b: int, F: dict) -> bool:
        ta, tb = set(_toks(F["_text"][a])), set(_toks(F["_text"][b]))
        return len(ta & tb) >= max(2, 0.25 * min(len(ta), len(tb)))

    def pick(self, t: int, cands: list[int], F: dict, prompt: str) -> int | None:
        p = F["partner"].get(str(t))
        L = max(1, len(cands))
        u = self._u(("p", t, tuple(sorted(cands))), prompt)
        if p is not None and p in cands and not self._unfindable(t, F):
            a, tau = FIND[F["bucket"].get(str(t), "mid")]
            if u < a * math.exp(-max(0, L - 10) / tau):
                return p
            wrong = WRONG["listed"]
        else:
            wrong = WRONG["absent_10"] if L <= 12 else WRONG["absent_long"]
        if rng_for("xabp-wrong", t, prompt).random() < wrong:
            others = [c for c in cands if c != p]
            if others:
                tt = set(_toks(F["_text"][t]))
                return max(others, key=lambda c: (len(tt & set(_toks(F["_text"][c]))), -c))
        return None

    def respond(self, prompt: str, hidden: dict, st: dict) -> str:
        F = dict(hidden["xab"])
        F["_text"] = _texts(F["tmap"])
        tmap = F["tmap"]
        lines = prompt.split("\n")
        items = []
        instr_lines = []
        for n, line in enumerate(lines):
            s = line.strip()
            iid, pre = _match_line(s, tmap) if len(s) >= 8 else (None, "")
            if iid is not None:
                lab = _LABEL.search(pre)
                items.append((n, iid, lab.group(1) if lab else str(iid), pre))
            else:
                instr_lines.append(line)
        instr = "\n".join(instr_lines)
        st["xab_items"] = st.get("xab_items", 0) + len(items)
        if not items:
            return self.chat(prompt, "I don't see any claims to compare.")
        pairs = []
        for k, (n, iid, _lab, s) in enumerate(items):
            m = _PAIR_A.match(s)
            if m and k + 1 < len(items) and _PAIR_B.match(items[k + 1][3]):
                pairs.append((m.group(1), iid, items[k + 1][1]))
        if pairs:
            st["xab_pair_calls"] = st.get("xab_pair_calls", 0) + 1
            out = {num: ("YES" if self.judge(a, b, F, prompt) else "NO") for num, a, b in pairs}
            return self.chat(prompt, json.dumps(out) if _wants_json(instr) else
                             "\n".join(f"{k}: {v}" for k, v in out.items()))
        if len(items) == 2 and re.search(r"(?i)\b(yes|no)\b", instr):
            st["xab_pair_calls"] = st.get("xab_pair_calls", 0) + 1
            return self.chat(prompt, "YES" if self.judge(items[0][1], items[1][1], F, prompt) else "NO")
        low = [ln.lower() for ln in lines]
        tgt_rows = [x for x in items if re.search(r"(?i)\btarget\b", x[3])]
        sect_t = next((n for n, ln in enumerate(low) if re.match(r"\s*targets?\s*:", ln)), None)
        sect_l = next((n for n, ln in enumerate(low) if re.match(r"\s*(list|candidates)\s*:", ln) and
                       (sect_t is None or n > sect_t)), None)
        plan = []
        if sect_t is not None and sect_l is not None and not tgt_rows:
            T = [x for x in items if sect_t < x[0] < sect_l]
            Cs = [x for x in items if x[0] > sect_l]
            for x in T:
                cand = [c for c in Cs if F["side"][str(c[1])] != F["side"][str(x[1])]]
                plan.append((x[1], x[2], cand))
        elif tgt_rows:
            starts = [x[0] for x in tgt_rows] + [len(lines)]
            for k, x in enumerate(tgt_rows):
                cand = [c for c in items if starts[k] < c[0] < starts[k + 1] and c not in tgt_rows
                        and F["side"][str(c[1])] != F["side"][str(x[1])]]
                plan.append((x[1], x[2], cand))
        if plan:
            st["xab_pick_calls"] = st.get("xab_pick_calls", 0) + 1
            st["xab_max_list"] = max(st.get("xab_max_list", 0), max(len(c) for _, _, c in plan))
            out = {}
            for t, lab, cand in plan:
                ids = [c[1] for c in cand]
                v = self.pick(t, ids, F, prompt)
                out[lab] = next((int(c[2]) if c[2].isdigit() else c[2] for c in cand if c[1] == v), "NONE") \
                    if v is not None else "NONE"
            return self.chat(prompt, json.dumps(out) if _wants_json(instr) else
                             "\n".join(f"{k}: {v}" for k, v in out.items()))
        st["xab_bulk_calls"] = st.get("xab_bulk_calls", 0) + 1
        shown = [x[1] for x in items]
        lab_of = {x[1]: x[2] for x in items}
        out = {}
        for x in items:
            cand = [c for c in shown if F["side"][str(c)] != F["side"][str(x[1])]]
            v = self.pick(x[1], cand, F, prompt)
            out[x[2]] = (int(lab_of[v]) if lab_of[v].isdigit() else lab_of[v]) if v is not None else "NONE"
        if re.search(r"(?i)unmatched|no paraphrase|orphan|missing|without a (?:match|paraphrase)", instr) and \
                not re.search(r"(?i)\bfor each\b", instr):
            un = [k for k, v in out.items() if v == "NONE"]
            body = json.dumps(un) if _wants_json(instr) else ", ".join(f"[{k}]" for k in un)
            return self.chat(prompt, body)
        return self.chat(prompt, json.dumps(out) if _wants_json(instr) else
                         "\n".join(f"{k}: {v}" for k, v in out.items()))


__all__ = ["FAMILY", "RUNG_PAIRS", "XAbsenceParaOracle", "compose", "facts", "pool"]
