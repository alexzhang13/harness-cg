"""Build ``sidecars/eval_index.npz``: hashes (no text) of every item in the published CTC eval set.

    python tools/build_eval_index.py [--workers 16]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "real" / "CTCBench"))

DATASET = "PrasannSinghal/ctc-suite-eval"
CACHE = os.environ.get("CTC_EVAL_CACHE")
TEMPLATE = {"textgroups", "strmatch"}
OUTLIER = {"outlier", "outlier_fixedM", "outlier_amzn"}
PUBMED = {"xabsence", "xabsence_paraphrase", "contradiction_iid"}
WORD = re.compile(r"[a-z0-9]+")


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=8).digest(), "little")


def norm(s: str) -> str:
    return " ".join((s or "").split())


def grams(s: str, n: int = 13) -> list[int]:
    w = WORD.findall((s or "").lower())
    return [h64(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)]


def row_key(docs) -> str:
    h = hashlib.sha1()
    for d in docs or []:
        h.update((d.get("text") or "").encode("utf-8", "ignore"))
        h.update(b"\x1e")
    return h.hexdigest()[:20]


def local_path(fname: str) -> str:
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(DATASET, fname, repo_type="dataset", local_files_only=True)
    except Exception:  # noqa: BLE001
        return hf_hub_download(DATASET, fname, repo_type="dataset", cache_dir=CACHE)


def gold_positions(ex: dict, subset: str) -> list[int] | None:
    """0-based document positions of the row's gold items, or None when the spec has no document gold."""
    from ctcbench_real import core as C
    from ctcbench_real import env as R
    try:
        spec = C.spec_of(subset)
    except Exception:  # noqa: BLE001
        return None
    n = len(ex.get("documents") or [])
    if spec.name in ("reorder", "grouping", "textgroups"):
        return list(range(n))
    ints = R._flat_ints(R._gold_of(ex, spec))
    base = spec.gold_index_base
    return [i - base for i in ints if 0 <= i - base < n]


def one_file(fname: str) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq
    subset = fname.split("/")[1]
    gold, gold13, outl, rows, pub13 = set(), set(), set(), set(), set()
    pf = pq.ParquetFile(local_path(fname))
    for batch in pf.iter_batches(batch_size=8):
        for ex in batch.to_pylist():
            docs = ex.get("documents") or []
            rows.add(int(row_key(docs), 16) & (2**64 - 1))
            texts = [norm(d.get("text")) for d in docs]
            gp = gold_positions(ex, subset)
            items = [texts[i] for i in (gp or [])] + [norm(q) for q in (ex.get("queries") or []) if isinstance(q, str)]
            for t in items:
                if t:
                    gold.add(h64(t))
                    if subset not in TEMPLATE:
                        gold13.update(grams(t))
            if subset in OUTLIER:
                outl.update(h64(t) for t in texts if t)
            if subset in PUBMED:
                for t in texts:
                    pub13.update(grams(t))
    arr = lambda s: np.fromiter(s, dtype=np.uint64, count=len(s))
    return {"gold": arr(gold), "gold13": arr(gold13), "outlier": arr(outl), "rows": arr(rows), "pubmed13": arr(pub13),
            "_file": fname}


def main() -> None:
    from huggingface_hub import HfApi
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(HERE.parent / "ctcbench_synth" / "sidecars" / "eval_index.npz"))
    a = ap.parse_args()
    files = sorted(f for f in HfApi().list_repo_files(DATASET, repo_type="dataset") if f.endswith(".parquet"))
    acc: dict[str, list[np.ndarray]] = {k: [] for k in ("gold", "gold13", "outlier", "rows", "pubmed13")}
    with ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(one_file, files):
            for k in acc:
                acc[k].append(r[k])
            print(f"{r['_file']}: gold {len(r['gold'])} outlier {len(r['outlier'])} pubmed13 {len(r['pubmed13'])}",
                  flush=True)
    out = {k: np.unique(np.concatenate(v)) for k, v in acc.items()}
    out["files"] = np.array(files)
    np.savez_compressed(a.out, **out)
    print("wrote", a.out, {k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
