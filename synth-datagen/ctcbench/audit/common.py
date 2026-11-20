"""Tokenisation and hashing for the CTC eval-overlap index: a hit is an exact normalised text match (`ekey`) or a
shared word 13-gram (`grams`). Run with PYTHONHASHSEED=0."""
import os, re, hashlib, json
import numpy as np
DATAGEN = os.environ.get("CTC_DATAGEN_DIR") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../outputs/synth-datagen/ctcbench"))
IDX = os.environ.get("CTC_AUDIT_INDEX") or os.path.join(DATAGEN, "audit", "idx")
SEED_POOLS = ("PrasannSinghal/ctc-seed-pools", "a2cbd4c1f0c900fbc8b2941f365b44e25baeead6")
SUITE_EVAL = ("PrasannSinghal/ctc-suite-eval", "04ab86006124b3f34c566f44bbc82d4fec98fcda")


def seed_pool(name):
    """Local path of one published CTC seed pool file (e.g. 'contradiction.seed.jsonl.gz'), at the revision used."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(SEED_POOLS[0], name, repo_type="dataset", revision=SEED_POOLS[1])


def eval_dir():
    """Local `data/` dir of ctc-suite-eval (<family>/<rung>.parquet), at the revision used."""
    from huggingface_hub import snapshot_download
    return os.path.join(snapshot_download(SUITE_EVAL[0], repo_type="dataset", revision=SUITE_EVAL[1], allow_patterns="data/*"), "data")


TOK = re.compile(r"[a-z0-9]+")
N = 13
_r = np.random.default_rng(13)
C = (_r.integers(1, 2 ** 62, size=N, dtype=np.uint64) | np.uint64(1))
M63 = (1 << 63) - 1

def toks(s):
    return TOK.findall(str(s).lower())

def norm(s):
    return " ".join(toks(s))

def ekey(s):
    """64-bit key of the normalised token string ('' -> None)."""
    n = norm(s)
    if not n:
        return None
    return int.from_bytes(hashlib.blake2b(n.encode(), digest_size=8).digest(), "little") & M63

_V = {}
def wid(x):
    v = _V.get(x)
    if v is None:
        v = _V[x] = (int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "little") & M63) | 1
    return v

def grams(s):
    w = toks(s)
    if len(w) < N:
        return np.empty(0, dtype=np.uint64)
    a = np.fromiter((wid(x) for x in w), dtype=np.uint64, count=len(w))
    m = len(a) - N + 1
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(N):
            h += a[k:k + m] * C[k]
    return h

def grams_text(s):
    w = toks(s)
    return [" ".join(w[i:i + N]) for i in range(len(w) - N + 1)]

class Index:
    """Loaded eval index: per family sorted uint64 gram arrays for gold-level / distractor-level, and exact-key dicts."""
    def __init__(self, fams=None):
        man = json.load(open(IDX + "/manifest.json"))
        self.fams = [f for f in man["fams"] if fams is None or f in fams]
        self.g = {}; self.e = {}
        for f in self.fams:
            for lv in ("gold", "dist"):
                self.g[(f, lv)] = np.load(f"{IDX}/{f}.{lv}.npy")
            self.e[f] = json.load(open(f"{IDX}/{f}.exact.json"))
            self.e[f] = {int(k): v for k, v in self.e[f].items()}
    def ghits(self, h):
        """{(fam,lv): bool mask over h}"""
        out = {}
        if len(h) == 0:
            return out
        for (f, lv), g in self.g.items():
            if len(g) == 0:
                continue
            i = np.searchsorted(g, h); i[i >= len(g)] = 0
            m = g[i] == h
            if m.any():
                out[(f, lv)] = m
        return out
    def exact(self, k):
        """{family: role bits} for an exact key (bit 1 = gold-level text, bit 2 = distractor-level text)."""
        if k is None:
            return {}
        return {f: d[k] for f, d in self.e.items() if k in d}
