"""Merged 13-gram and exact-key arrays over all 23 eval families, for the outlier row builder.

    python all23.py build | check <rows.jsonl[.gz]> ..."""
import gzip, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audit"))
import common as A  # noqa: E402
OUT = os.path.join(A.DATAGEN, "outlier", "all23")


def build():
    man = json.load(open(A.IDX + "/manifest.json"))
    gs, ks = [], set()
    for f in man["fams"]:
        for lv in ("gold", "dist"):
            gs.append(np.load(f"{A.IDX}/{f}.{lv}.npy"))
        ks |= {int(k) for k in json.load(open(f"{A.IDX}/{f}.exact.json"))}
    g = np.unique(np.concatenate(gs)); e = np.array(sorted(ks), dtype=np.uint64)
    os.makedirs(OUT, exist_ok=True)
    np.save(OUT + "/grams.npy", g); np.save(OUT + "/exact.npy", e)
    print("families", len(man["fams"]), "13-grams", len(g), "exact keys", len(e))


def load():
    return np.load(OUT + "/grams.npy"), np.load(OUT + "/exact.npy")


def _in(arr, h):
    if not len(h):
        return np.zeros(0, bool)
    i = np.searchsorted(arr, h); i[i >= len(arr)] = 0
    return arr[i] == h


def dirty(text, G, X):
    k = A.ekey(text)
    if k is not None and _in(X, np.array([k], dtype=np.uint64))[0]:
        return True
    return bool(_in(G, A.grams(text)).any())


def check(paths):
    G, X = load(); tot = bad = 0
    for p in paths:
        with (gzip.open if p.endswith(".gz") else open)(p, "rt") as f:
            for k, line in enumerate(f):
                ex = json.loads(line)
                n = sum(dirty(d["text"], G, X) for d in ex["documents"])
                tot += 1; bad += n > 0
                print(json.dumps({"file": p, "row": k, "dirty_passages": n}), flush=True)
    print(json.dumps({"summary": {"rows": tot, "contaminated": bad}}), flush=True)


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else check(sys.argv[2:])
