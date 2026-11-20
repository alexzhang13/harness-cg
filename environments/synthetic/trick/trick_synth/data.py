"""Problem corpus and lookup tables for the oracle.

The corpus (125 techniques, 5,054 problems, no OBLIQ text) was written by Qwen3.8-27B with tools/gen_corpus.py and
cleaned up by the other tools/ scripts. 20% of techniques form the val split.
"""
from __future__ import annotations
import json, os, re, random
from functools import lru_cache
from rlm_train.frozen import frozen_path

_F = os.path.join(os.path.dirname(__file__), "data", os.environ.get("TRICK_CORPUS", "corpus.json"))
VAL_FRAC = 0.2

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

@lru_cache(maxsize=1)
def corpus() -> dict:
    C = json.load(open(frozen_path(_F, "trick/" + os.path.basename(_F))))
    T, I = C["techniques"], [it for it in C["items"] if it.get("keep", True)]
    rng = random.Random(7)
    ids = list(range(len(T))); rng.shuffle(ids)
    val = set(ids[: int(len(T) * VAL_FRAC)])
    by_tech: dict[int, list[int]] = {}
    for k, it in enumerate(I):
        by_tech.setdefault(it["tech"], []).append(k)
    look: dict[str, int] = {}
    clash: set[str] = set()
    for t in T:
        for fp in t["fingerprints"] + [t["desc"]]:
            k = _norm(fp)[:48]
            if k in look and look[k] != t["id"]:
                clash.add(k)
            look[k] = t["id"]
    for k in clash:
        look.pop(k, None)
    sk: dict[str, int] = {}
    for it in I:
        sk[_norm(it["sketch"])] = it["tech"]
    fam = {t["id"]: t["family"] for t in T}
    fam_members: dict[str, list[int]] = {}
    for t in T:
        fam_members.setdefault(t["family"], []).append(t["id"])
    return {"T": T, "I": I, "val": val, "by_tech": by_tech, "look": look, "sk": sk, "fam": fam, "fam_members": fam_members}

def lookup_desc(line_norm: str) -> int | None:
    """Technique named by a line carrying one of our fingerprints, else an exact item sketch."""
    C = corpus(); L, SK = C["look"], C["sk"]
    segs = [seg.strip(" -:*[]0123456789.").strip() for seg in re.split(r"\||meta:|idea:|sketch:|fingerprint:|trick:", line_norm)]
    for s in segs:
        if len(s) >= 30 and s[:48] in L:
            return L[s[:48]]
    for s in segs:
        if len(s) >= 30 and s in SK:
            return SK[s]
    return None

def family(t: int) -> str:
    return corpus()["fam"][t]

def confuse(t: int, r: random.Random) -> int:
    C = corpus()
    if r.random() < 0.6:
        cands = [x for x in C["fam_members"][C["fam"][t]] if x != t]
        if cands:
            return r.choice(cands)
    return r.randrange(len(C["T"]))

def fingerprint(t: int, r: random.Random) -> str:
    fps = corpus()["T"][t]["fingerprints"]
    return r.choice(fps)

def random_sketch(t: int, r: random.Random) -> str:
    C = corpus(); k = r.choice(C["by_tech"][t])
    return C["I"][k]["sketch"]

def sketch_for_query(qiid: int, t: int) -> str:
    C = corpus(); it = C["I"][qiid]
    return it["sketch"] if it["tech"] == t else random_sketch(t, random.Random(qiid))


_WR = re.compile(r"[a-z]{4,}")

def closest_technique(prompt: str, hidden: dict) -> int | None:
    """Technique a describe request is about: the query's if enough of it is present, else the best word overlap."""
    pw = set(_WR.findall(prompt.lower()))
    qw = set(hidden.get("_qwords") or [])
    if qw and len(qw & pw) >= max(3, 0.3 * len(qw)):
        return hidden["qtech"]
    best, bs = None, 0
    for t in corpus()["T"]:
        tw = set(_WR.findall((t["name"] + " " + t["desc"] + " " + " ".join(t["fingerprints"])).lower()))
        sc = len(tw & pw) / (1 + len(tw) ** 0.5)
        if sc > bs:
            best, bs = t["id"], sc
    return best if bs >= 0.5 else None
