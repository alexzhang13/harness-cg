"""outlier / outlier_fixedM: oracle answers to the two JSON protocols of the fixedM seam program."""

from __future__ import annotations

import json
import re

from synth_common.oracle import rng_for

P_TITLE = 0.58
P_SEAM_HIT = 0.887
P_SEAM_FALSE = (0.68, 0.88)
P_MEMBER_HIT = 0.81
P_MEMBER_FP = 0.007


def protocol_answer(oracle, view, hidden: dict, st: dict, fragment_label, bump):
    low = view.low
    f = hidden
    if "art" not in f or "chunk" not in f:
        return None
    if "whose beginning is the direct continuation" in low:
        out = {}
        for m in re.finditer(r"(?s)QUERY (\d+) ends:.*?(?=\n\nQUERY |\n\nAnswer with JSON|\Z)", view.prompt):
            q = m.group(1)
            cands = re.findall(r"(?m)^\s*candidate (\d+):", m.group(0))
            if q not in f["art"]:
                continue
            truth = next((c for c in cands if c in f["art"] and f["art"][c] == f["art"][q]
                          and f["chunk"][c] == f["chunk"][q] + 1), None)
            r = rng_for(view.prompt, q, "seam-choice").random()
            if truth and r < P_SEAM_HIT - oracle.err_prob(view.prompt):
                out[q] = int(truth)
            elif cands and r < P_SEAM_FALSE[len(f.get("items") or []) > 400]:
                same = [c for c in cands if c != truth and c in f["art"] and f["art"][c] == f["art"][q]]
                pool = same or [c for c in cands if c != truth] or cands
                out[q] = int(rng_for(view.prompt, q, "seam-false").choice(pool))
            else:
                out[q] = 0
        bump(st, "seam_choice_calls")
        return json.dumps(out)
    if "cut out of the same wikipedia article as the group" in low:
        groups = re.findall(r"(?m)^GROUP passage (\d+):", view.prompt)
        cands = re.findall(r"(?m)^CANDIDATE (\d+):", view.prompt)
        arts = {f["art"].get(g) for g in groups}
        named = []
        for c in cands:
            same = len(arts) == 1 and f["art"].get(c) in arts
            if rng_for(view.prompt, c, "member").random() < (P_MEMBER_HIT if same else P_MEMBER_FP):
                named.append(int(c))
        bump(st, "member_calls")
        return json.dumps({"same_article": named})
    if "name the wikipedia article it was most likely cut from" in low:
        texts = {str(i): str(t) for i, t in f.get("items") or []}
        rows = []
        for m in re.finditer(r"(?m)^Passage (\d+): ", view.prompt):
            i = m.group(1)
            if i not in f["art"]:
                continue
            title = f["title"][str(f["art"][i])]
            ok = rng_for(view.prompt, i, "title").random() < P_TITLE - oracle.err_prob(view.prompt)
            frag = fragment_label(texts.get(i, ""), title)
            alt = f"{frag} {title}" if rng_for(view.prompt, i, "title-alt").random() < 0.5 else frag
            t = title if ok else alt
            rows.append({"id": int(i), "title": t, "topic": t})
        bump(st, "title_calls")
        return json.dumps(rows)
    return None
