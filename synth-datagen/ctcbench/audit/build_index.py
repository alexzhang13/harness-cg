"""Index every published ctc-suite-eval parquet: exact text keys and 13-gram hashes, split into gold-level and
distractor-level. Writes into $CTC_AUDIT_INDEX.

    PYTHONHASHSEED=0 python build_index.py [workers=32]
"""
import json, os, sys, glob
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import pyarrow.parquet as pq

BASE = {"qdmatch_fiqa": 1, "qdmatch_nq": 1, "qdmatch_hpqa": 1, "reorder": 1, "contradiction_iid": 1, "strmatch": 1,
        "textgroups": 1}
ALLGOLD = {"grouping", "reorder"}

def flat(x):
    if isinstance(x, str):
        try: x = json.loads(x)
        except Exception: return []
    if isinstance(x, (list, tuple)): return [i for v in x for i in flat(v)]
    try: return [int(x)]
    except Exception: return []

def work(path):
    fam = path.split("/")[-2]; rung = path.split("/")[-1][:-8]
    b = BASE.get(fam, 0)
    t = pq.read_table(path)
    cols = set(t.column_names)
    ex_bits, gold_txt, dist_txt = {}, set(), set()
    nrows = t.num_rows
    for ex in t.to_pylist():
        docs = ex["documents"] or []
        if fam in ALLGOLD:
            gi = set(range(len(docs)))
        elif "gold_pairs" in cols:
            gi = {i - b for i in flat(ex.get("gold_pairs"))}
        else:
            gi = {i - b for i in flat(ex.get("gold_doc_indices"))}
        for i, d in enumerate(docs):
            tx = d["text"] if isinstance(d, dict) else str(d)
            (gold_txt if i in gi else dist_txt).add(tx)
        q = ex.get("queries")
        if isinstance(q, str):
            try: q = json.loads(q)
            except Exception: q = [q]
        for x in (q or []):
            if fam not in ("absence_gutenberg",):   # its queries[0] is the corpus itself (distractor-level)
                gold_txt.add(str(x))
            else:
                dist_txt.add(str(x))
    dist_txt -= gold_txt
    for bit, S in ((1, gold_txt), (2, dist_txt)):
        for tx in S:
            k = ekey(tx)
            if k is not None: ex_bits[k] = ex_bits.get(k, 0) | bit
    gg = np.unique(np.concatenate([grams(x) for x in gold_txt] + [np.empty(0, np.uint64)]))
    dg = np.unique(np.concatenate([grams(x) for x in dist_txt] + [np.empty(0, np.uint64)]))
    np.save(f"{IDX}/part.{fam}.{rung}.gold.npy", gg); np.save(f"{IDX}/part.{fam}.{rung}.dist.npy", dg)
    json.dump(ex_bits, open(f"{IDX}/part.{fam}.{rung}.exact.json", "w"))
    return fam, rung, nrows, len(gold_txt), len(dist_txt), len(gg), len(dg)

if __name__ == "__main__":
    os.makedirs(IDX, exist_ok=True)
    files = sorted(glob.glob(eval_dir() + "/*/*.parquet"), key=lambda p: -os.path.getsize(p))
    stats = []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 32) as P:
        for r in P.imap_unordered(work, files):
            print(*r, flush=True); stats.append(r)
    fams = sorted({s[0] for s in stats})
    for f in fams:
        for lv in ("gold", "dist"):
            parts = [np.load(p) for p in glob.glob(f"{IDX}/part.{f}.*.{lv}.npy")]
            np.save(f"{IDX}/{f}.{lv}.npy", np.unique(np.concatenate(parts)))
        e = {}
        for p in glob.glob(f"{IDX}/part.{f}.*.exact.json"):
            for k, v in json.load(open(p)).items(): e[k] = e.get(k, 0) | v
        json.dump(e, open(f"{IDX}/{f}.exact.json", "w"))
        print("merged", f, len(e), flush=True)
    json.dump({"fams": fams, "files": [list(s) for s in stats]}, open(f"{IDX}/manifest.json", "w"), indent=1)
    for p in glob.glob(f"{IDX}/part.*"): os.remove(p)
    print("DONE")
