"""Build the fact sidecars in ctcbench_synth/sidecars/ from the CTC seed pools (one-time, offline).

    python build_sidecars.py [outlier] [reorder]

outlier: source article and chunk index of every outlier document. reorder: seam positions of every reorder row.
"""

from __future__ import annotations

import codecs
import collections
import glob
import gzip
import json
import os
import statistics
import sys

os.environ["CTCBENCH_SIDECAR_BUILD"] = "1"
import time
from pathlib import Path

from ctcbench_synth.data import SIDECARS, examples, meta, row_key, text_key

POOLS = Path(glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--PrasannSinghal--ctc-seed-pools/snapshots/*"))[0])
EVAL = Path(glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--PrasannSinghal--ctc-suite-eval/snapshots/*/data"))[0])


def _stream_array(path: Path, key: str):
    """Yield the elements of the top-level JSON array ``key`` on the second line of a gz pool, incrementally."""
    dec = json.JSONDecoder()
    udec = codecs.getincrementaldecoder("utf-8")("ignore")
    buf, pos, started = "", 0, False
    marker = f'"{key}": ['
    with gzip.open(path, "rb") as f:
        f.readline()
        while True:
            block = f.read(1 << 24)
            if block:
                buf = buf[pos:] + udec.decode(block)
                pos = 0
            if not started:
                i = buf.find(marker)
                if i < 0:
                    if not block:
                        return
                    continue
                pos, started = i + len(marker), True
            while True:
                while pos < len(buf) and buf[pos] in " \n\r\t,":
                    pos += 1
                if pos >= len(buf):
                    break
                if buf[pos] == "]":
                    return
                try:
                    obj, end = dec.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    break
                yield obj
                pos = end
            if not block:
                return


def build_outlier() -> None:
    need: dict[str, list] = {}
    rows = {}
    for fam in ("outlier", "outlier_fixedM"):
        for split in ("train", "val"):
            for ex in examples(fam, split):
                rk = row_key(ex)
                rows[rk] = (fam, split, len(ex["documents"]))
                for j, d in enumerate(ex["documents"]):
                    need.setdefault(text_key(d["text"]), []).append((rk, j))
    eval_need: dict[str, list] = collections.defaultdict(list)
    import pyarrow.parquet as pq
    eval_rows = []
    for fam in ("outlier", "outlier_fixedM"):
        for p in sorted((EVAL / fam).glob("*.parquet")):
            for ri, ex in enumerate(pq.read_table(p, columns=["documents", "gold_doc_indices"]).to_pylist()):
                eid = f"{fam}/{p.stem}/{ri}"
                eval_rows.append((eid, len(ex["documents"]), ex["gold_doc_indices"]))
                for j, d in enumerate(ex["documents"]):
                    eval_need[text_key(d["text"])].append((eid, j))
    print(f"outlier: {len(rows)} train/val rows, {len(need)} distinct chunk texts; eval rows {len(eval_rows)}, "
          f"{len(eval_need)} distinct eval chunk texts", flush=True)
    titles: list[str] = []
    tid: dict[str, int] = {}
    nchunks: list[int] = []
    found: dict[tuple, tuple] = {}
    eval_found: dict[tuple, int] = {}
    t0 = time.time()
    n_art = 0
    for art in _stream_array(POOLS / "outlier.seed.jsonl.gz", "articles"):
        n_art += 1
        title, chunks = art[0], art[1]
        hit_tr = [(j, need[h]) for j, h in enumerate(map(text_key, chunks)) if h in need]
        hit_ev = [(j, eval_need[h]) for j, h in enumerate(map(text_key, chunks)) if h in eval_need] if eval_need else []
        if hit_tr or hit_ev:
            if title not in tid:
                tid[title] = len(titles)
                titles.append(title)
                nchunks.append(len(chunks))
            a = tid[title]
            for j, locs in hit_tr:
                for rk, dj in locs:
                    found[(rk, dj)] = (a, j)
            for _j, locs in hit_ev:
                for eid, dj in locs:
                    eval_found[(eid, dj)] = a
        if n_art % 500000 == 0:
            print(f"  {n_art} articles, {time.time() - t0:.0f}s, mapped {len(found)} train docs", flush=True)
    print(f"pool: {n_art} articles scanned in {time.time() - t0:.0f}s", flush=True)
    out_rows = {}
    miss = 0
    for rk, (_fam, _split, n) in rows.items():
        lab = []
        for j in range(n):
            v = found.get((rk, j))
            if v is None:
                miss += 1
                lab.append([-1, -1])
            else:
                lab.append(list(v))
        out_rows[rk] = lab
    print(f"train/val docs unmapped: {miss}")
    train_titles = {a for lab in out_rows.values() for a, _ in lab if a >= 0}
    eval_titles = set(eval_found.values())
    print(f"articles: train/val {len(train_titles)}, eval {len(eval_titles)}, overlap {len(train_titles & eval_titles)}")
    by_rung = collections.defaultdict(list)
    for eid, n, gold in eval_rows:
        labs = [eval_found.get((eid, j), -1) for j in range(n)]
        c = collections.Counter(labs)
        gold_art = {labs[g] for g in gold}
        others = [v for k, v in c.items() if k not in gold_art and k >= 0]
        by_rung[eid.rsplit("/", 1)[0]].append((min(others) if others else 0, sum(1 for v in others if v <= 3),
                                              len(c), c.get(-1, 0), len(gold_art)))
    for rung, v in sorted(by_rung.items()):
        print(f"  eval {rung}: rows {len(v)}, min other-article size med {statistics.median(x[0] for x in v)}, "
              f"rows with another article <=3 chunks {sum(1 for x in v if x[1])}, articles/row med "
              f"{statistics.median(x[2] for x in v)}, unmapped docs/row max {max(x[3] for x in v)}, "
              f"gold articles/row {collections.Counter(x[4] for x in v)}")
    SIDECARS.mkdir(parents=True, exist_ok=True)
    with gzip.open(SIDECARS / "outlier_articles.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"titles": titles, "nchunks": nchunks, "rows": out_rows,
                   "eval_articles": sorted(eval_titles & train_titles)}, f)
    print("wrote", SIDECARS / "outlier_articles.json.gz")


def build_reorder() -> None:
    with gzip.open(POOLS / "reorder.seed.jsonl.gz", "rt", encoding="utf-8") as f:
        f.readline()
        pool = json.loads(f.readline())
    books = {b["book"]: b for b in pool["books"]}
    out, bad = {}, 0
    for split in ("train", "val"):
        for ex in examples("reorder", split):
            m = meta(ex)
            b = books.get(m.get("book"))
            n, start = int(m["n"]), int(m["start"])
            order = [int(i) for i in ex["gold_order"]]
            docs = ex["documents"]
            if b is None:
                bad += 1
                continue
            src = b["passages"][start:start + n]
            ok = all(" ".join(docs[order[p] - 1]["text"].split()) == " ".join(src[p].split()) for p in range(n))
            seams = sorted(s - start for s in b["seams"] if start < s < start + n)
            if not ok or len(seams) != int(m.get("run_seams", len(seams))):
                bad += 1
            out[row_key(ex)] = seams
    SIDECARS.mkdir(parents=True, exist_ok=True)
    (SIDECARS / "reorder_seams.json").write_text(json.dumps(out))
    print(f"reorder: {len(out)} rows, {bad} mismatched; wrote reorder_seams.json")


if __name__ == "__main__":
    which = sys.argv[1:] or ["reorder", "outlier"]
    for w in which:
        {"outlier": build_outlier, "reorder": build_reorder}[w]()
