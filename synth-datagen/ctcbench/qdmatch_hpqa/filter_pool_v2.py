"""Drop qdmatch_hpqa seed-pool units whose text matches any eval item of the 23 families (exact or 13-gram).

Uses the audit index ($CTC_AUDIT_INDEX) and also applies decontam_strict.banned. SRC '-' means the published pool.

    PYTHONHASHSEED=0 python filter_pool_v2.py SRC.seed.jsonl.gz OUT_DIR/qdmatch_hpqa.seed.jsonl.gz REPORT.json
"""
import gzip
import json
import os
import sys
from pathlib import Path

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audit"), str(Path(__file__).resolve().parent)]
import common  # noqa: E402
from decontam_strict import banned, grams as sgrams, norm as snorm  # noqa: E402


def unit_texts(u: dict) -> list[str]:
    return [u.get("query", "")] + [str(x) for v in u.values() if isinstance(v, list) for x in v]


def main(src: str, out: str, rep: str) -> None:
    if src == "-":
        src = common.seed_pool("qdmatch_hpqa.seed.jsonl.gz")
    ix = common.Index()
    bq, bd, bg, _ = banned()
    f = gzip.open(src, "rt")
    header = json.loads(f.readline())
    pool = json.loads(f.readline())
    keep, why = [], {"exact": 0, "13gram": 0, "strict_qd_hp": 0}
    fams = {}
    for u in pool["units"]:
        bad = None
        for t in unit_texts(u):
            ex = ix.exact(common.ekey(t))
            if ex:
                bad = "exact"; [fams.__setitem__(k, fams.get(k, 0) + 1) for k in ex]
                break
            gh = ix.ghits(common.grams(t))
            if gh:
                bad = "13gram"; [fams.__setitem__(k[0], fams.get(k[0], 0) + 1) for k in gh]
                break
            n = snorm(t)
            if n in bq or n in bd or (sgrams(n) & bg):
                bad = "strict_qd_hp"
                break
        if bad:
            why[bad] += 1
        else:
            keep.append(u)
    pool["units"] = keep
    pool.setdefault("provenance", {})["decontam_23fam_2026_10_22"] = {"kept": len(keep), "dropped": why}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as g:
        g.write(json.dumps(header) + "\n")
        g.write(json.dumps(pool) + "\n")
    r = {"src_units": len(keep) + sum(why.values()), "kept": len(keep), "dropped": why, "first_hit_family_counts": fams}
    Path(rep).write_text(json.dumps(r, indent=1))
    print(r)


if __name__ == "__main__":
    main(*sys.argv[1:4])
