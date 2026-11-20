"""Check that built qdmatch rows have no eval hits (23 families, exact or 13-gram) and write normalised rows for
run_dev.py. Rows sharing gold with EXCLUDE_GOLD_FROM are dropped.

    PYTHONHASHSEED=0 python verify_23fam.py BUILT_train.jsonl OUT_rows.jsonl REPORT.json [EXCLUDE_GOLD_FROM.jsonl]
"""
import json
import os
import sys
from pathlib import Path

_REPO = os.path.abspath(__file__ + "/../../../..")
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audit"), _REPO + "/environments/real/CTCBench"]
import common  # noqa: E402
from ctcbench_real import env as E  # noqa: E402


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def gold_units(ex: dict) -> set:
    docs = ex["documents"]
    out = set()
    for q, d in ex["gold_pairs"]:
        for i in (q, d):
            out.add(_norm(docs[i - 1]["text"]) if isinstance(docs[i - 1], dict) else _norm(docs[i - 1]))
    return out


def main(src, out, rep, excl=None):
    ix = common.Index()
    ex_gold = set().union(*(gold_units(json.loads(l)) for l in open(excl))) if excl else set()
    n = kept = hit_rows = overlap = 0
    hits = []
    with open(out, "w") as f:
        for k, line in enumerate(open(src)):
            ex = E._normalize(json.loads(line)); n += 1
            if ex_gold and gold_units(ex) & ex_gold:
                overlap += 1
                continue
            h = 0
            for j, d in enumerate(ex["documents"]):
                t = d["text"] if isinstance(d, dict) else str(d)
                if ix.exact(common.ekey(t)) or ix.ghits(common.grams(t)):
                    h += 1
                    if len(hits) < 20:
                        hits.append({"row": k, "item": j, "text": t[:120]})
            if h:
                hit_rows += 1
                continue
            ex["_gen_row"] = k
            f.write(json.dumps(ex) + "\n"); kept += 1
    r = {"built": n, "gold_overlap_dropped": overlap, "rows_with_hits": hit_rows, "kept_clean": kept, "sample_hits": hits}
    Path(rep).write_text(json.dumps(r, indent=1))
    print({k: v for k, v in r.items() if k != "sample_hits"})


if __name__ == "__main__":
    main(*sys.argv[1:])
