"""qdmatch_hpqa: oracle answers to the three JSON evidence protocols of the QDMatch program."""

from __future__ import annotations

import json
import re

from synth_common.oracle import rng_for

from .oracle_base import Seen, View

P_SCREEN_GOLD = 0.964
P_SCREEN_FP = (0.007, 0.030)
P_VERIFY_GOLD = 0.966
P_VERIFY_FP = 0.038
P_ELLIPSIS = 0.10
P_CONFIRM_GOLD = 0.982
P_CONFIRM_FP = 0.50


def _sentence(text: str, words: set) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20] or [text.strip()]
    toks = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))  # noqa: E731
    return max(sents, key=lambda s: len(toks(s) & words))[:300]


def _quote(text: str, words: set, r: float) -> str:
    s = _sentence(text, words)
    w = s.split()
    if r < P_ELLIPSIS and len(w) >= 10:
        return " ".join(w[:4]) + " ... " + " ".join(w[-4:])
    return s


def protocol_answer(oracle, view: View, hidden: dict, st: dict) -> str | None:
    instr = view.instr
    f = hidden
    qset = {str(q) for q in f["queries"]}
    qs = [s for s in view.items if s.iid in qset]
    ds = [s for s in view.items if s.iid not in qset]
    from .oracle_base import texts_of
    alltexts = texts_of(hidden)
    texts = {s.iid: alltexts.get(s.iid) or view.prompt[s.pos:s.end] for s in view.items}
    if '"matches"' in instr and "bridge" in instr:
        return _screen(view, f, qs, ds, texts)
    if '"answerable"' in instr and len(qs) == 1:
        return _verify(view, f, qs[0], ds, texts)
    if '"supported"' in instr and len(qs) == 1:
        return _confirm(view, f, qs[0], ds)
    return None


def _screen_fp(f: dict) -> float:
    lo, hi = P_SCREEN_FP
    t = min(1.0, max(0.0, (len(f["queries"]) - 178) / (872 - 178)))
    return lo + (hi - lo) * t


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _screen(view: View, f: dict, qs: list[Seen], ds: list[Seen], texts: dict) -> str:
    out = []
    for q in qs:
        gold = set(f["rel"].get(q.iid) or [])
        g_here = [s for s in ds if int(s.iid) in gold]
        r = rng_for(view.prompt, q.iid, "screen").random()
        words = _words(texts[q.iid])
        if g_here and r < P_SCREEN_GOLD:
            use = g_here[:2]
            missing = [d for d in gold if str(d) not in {s.iid for s in ds}]
            bridge = (f["ent"].get(str(missing[0])) or "") if missing else ""
        elif not g_here and r < _screen_fp(f) and ds:
            rr = rng_for(view.prompt, q.iid, "fp")
            use = [max(ds, key=lambda s: len(_words(texts[s.iid]) & words) + rr.random())]
            bridge = f["ent"].get(use[0].iid) or ""
        else:
            continue
        out.append({"q": int(q.iid), "docs": [int(s.iid) for s in use],
                    "quote": _sentence(texts[use[0].iid], words), "bridge": bridge})
    return json.dumps({"matches": out}, ensure_ascii=False)


def _verify(view: View, f: dict, q: Seen, ds: list[Seen], texts: dict) -> str:
    gold = set(f["rel"].get(q.iid) or [])
    g_here = [s for s in ds if int(s.iid) in gold]
    r = rng_for(view.prompt, q.iid, "verify").random()
    words = _words(texts[q.iid])
    if len(g_here) == 2 and r < P_VERIFY_GOLD:
        use = g_here
    elif len(g_here) < 2 and r < P_VERIFY_FP and len(ds) >= 2:
        rr = rng_for(view.prompt, q.iid, "vfp")
        use = sorted(ds, key=lambda s: -(len(_words(texts[s.iid]) & words) + rr.random()))[:2]
    else:
        return json.dumps({"answerable": False})
    quotes = [_quote(texts[s.iid], words, rng_for(view.prompt, s.iid, "ell").random()) for s in use]
    return json.dumps({"answerable": True, "docs": [int(s.iid) for s in use], "quotes": quotes,
                       "answer": "(answer)", "connection": "the two passages together answer the question"},
                      ensure_ascii=False)


def _confirm(view: View, f: dict, q: Seen, ds: list[Seen]) -> str:
    gold = set(f["rel"].get(q.iid) or [])
    true_chain = len(ds) == 2 and all(int(s.iid) in gold for s in ds)
    r = rng_for(view.prompt, q.iid, "confirm").random()
    ok = r < (P_CONFIRM_GOLD if true_chain else P_CONFIRM_FP)
    return json.dumps({"clauses": [{"clause": "(question)", "passage": "A" if ok else "none"}], "supported": ok})
