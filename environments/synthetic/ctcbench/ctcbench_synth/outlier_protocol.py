"""outlier: oracle answers to the group-membership question of the outlier program."""

from __future__ import annotations

import json
import re

from synth_common.oracle import rng_for

P_SAME_TRUE = 0.996
P_SAME_MIXED = 0.44
P_OTHER_HIT = 0.84
P_OTHER_FP = 0.003
_DOC = re.compile(r"(?m)^Document \[(\d+)\]:")
P_FWD_HIT, P_FWD_NONE_TRUE, P_FWD_NONE_ABSENT = 0.86, 0.03, 0.41
P_BWD_HIT, P_BWD_NONE_TRUE, P_BWD_NONE_ABSENT = 0.67, 0.03, 0.24
P_VERIFY_TRUE, P_VERIFY_FALSE = 0.89, 0.035
_LETTERS = [chr(c) for c in range(ord("a"), ord("z") + 1)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def _keys(hidden):
    """{first 8 words: id}, {last 8 words: id} of every passage (cached on the payload)."""
    k = hidden.get("_outlier_seam_keys")
    if k is None:
        heads, tails = {}, {}
        for i, t in hidden.get("items") or []:
            w = str(t).split()
            heads.setdefault(" ".join(w[:8]), str(i))
            tails.setdefault(" ".join(w[-8:]), str(i))
        k = hidden["_outlier_seam_keys"] = (heads, tails)
    return k


def _head_id(hidden, frag):
    return _keys(hidden)[0].get(" ".join(frag.replace("...", " ").split()[:8]))


def _tail_id(hidden, frag):
    return _keys(hidden)[1].get(" ".join(frag.replace("...", " ").split()[-8:]))


def _next(f, a, b):
    return a in f["art"] and b in f["art"] and f["art"][a] == f["art"][b] and f["chunk"][b] == f["chunk"][a] + 1


def _seam_answer(view, f, st, bump):
    p, low = view.prompt, view.low
    if "each pair below shows the end of fragment a and the start of fragment b" in low:
        out = {}
        for m in re.finditer(r"(?s)PAIR (\d+)\nEND OF A: (.*?)\nSTART OF B: (.*?)(?=\n\nPAIR |\n\nAnswer with|\Z)", p):
            a, b = _tail_id(f, m.group(2)), _head_id(f, m.group(3))
            truth = a is not None and b is not None and _next(f, a, b)
            yes = rng_for(p, m.group(1), "outlier-verify").random() < (P_VERIFY_TRUE if truth else P_VERIFY_FALSE)
            out[m.group(1)] = "yes" if yes else "no"
        bump(st, "outlier_verify_calls")
        return json.dumps(out)
    fwd = "each item below shows the last words of a text fragment" in low
    bwd = "each item below shows the opening words of a text fragment" in low
    if not (fwd or bwd):
        return None
    out = {}
    for m in re.finditer(r"(?s)ITEM (\d+)\n(?:LAST|OPENING) WORDS: (.*?)\nCANDIDATES:\n(.*?)(?=\n\nITEM |\n\nAnswer with|\Z)", p):
        q = _tail_id(f, m.group(2)) if fwd else _head_id(f, m.group(2))
        cands = [(c.group(1), (_head_id(f, c.group(2)) if fwd else _tail_id(f, c.group(2))))
                 for c in re.finditer(r"(?m)^\s*\(([A-Za-z])\) (.*)$", m.group(3))]
        true = next((L for L, i in cands if q is not None and i is not None and
                     (_next(f, q, i) if fwd else _next(f, i, q))), None)
        r = rng_for(p, m.group(1), "outlier-pick").random()
        hit, none_t, none_a = (P_FWD_HIT, P_FWD_NONE_TRUE, P_FWD_NONE_ABSENT) if fwd else \
            (P_BWD_HIT, P_BWD_NONE_TRUE, P_BWD_NONE_ABSENT)
        if true is not None:
            if r < hit:
                out[m.group(1)] = true
                continue
            if r < hit + none_t or len(cands) < 2:
                out[m.group(1)] = "none"
                continue
        elif r < none_a or not cands:
            out[m.group(1)] = "none"
            continue
        same = [L for L, i in cands if L != true and i is not None and q is not None and i in f["art"]
                and q in f["art"] and f["art"][i] == f["art"][q]]
        pool = same or [L for L, _ in cands if L != true]
        out[m.group(1)] = rng_for(p, m.group(1), "outlier-pick-wrong").choice(pool)
    bump(st, "outlier_pick_calls")
    return json.dumps(out)


def protocol_answer(oracle, view, hidden: dict, st: dict, bump):
    f = hidden
    if "art" not in f or "chunk" not in f:
        return None
    if "were the three group passages all cut from one and the same wikipedia article" not in view.low:
        return _seam_answer(view, f, st, bump)
    p = view.prompt
    g0, o0 = p.find("GROUP:"), p.find("OTHER PASSAGES:")
    if g0 < 0 or o0 < 0:
        return None
    group = [i for i in _DOC.findall(p[g0:o0]) if i in f["art"]]
    others = [i for i in _DOC.findall(p[o0:]) if i in f["art"]]
    arts = {f["art"][i] for i in group}
    one = len(group) == 3 and len(arts) == 1
    same = rng_for(p, "outlier-member-same").random() < (P_SAME_TRUE if one else P_SAME_MIXED)
    named = []
    for j in others:
        hit = f["art"][j] in arts if one else any(f["art"][j] == f["art"][i] for i in group)
        if rng_for(p, j, "outlier-member-other").random() < (P_OTHER_HIT if hit else P_OTHER_FP):
            named.append(int(j))
    bump(st, "outlier_member_calls")
    return json.dumps({"same": bool(same), "others": named})
