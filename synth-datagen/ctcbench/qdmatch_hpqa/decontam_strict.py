"""Strict decontamination of generated qdmatch rows (qdmatch_hpqa / hotpotqa eval rungs).

A generated qdmatch row is contaminated if ANY of its items (queries or documents, relevant or distractor):
  - exactly matches (normalized text) an eval RELEVANT QUERY or an eval GOLD DOCUMENT from any published rung of
    qdmatch_hpqa (r2k..r256k) or hotpotqa (r2k..r1m), or
  - is a query sharing any 13-token n-gram with an eval relevant query.
Eval relevant queries: qdmatch_hpqa gold_pairs query items; hotpotqa queries[0].
Eval gold documents:   qdmatch_hpqa gold_pairs document items; hotpotqa gold_doc_indices (0-based).

  python decontam_strict.py OUT_REPORT.json ROWS.jsonl [ROWS.jsonl ...]
Report: per file, contaminated row positions (0-based line index) + _gen_row + reasons. The second paragraph of this
docstring is copied into the report as its "rule". `banned()` is also the second check of filter_pool_v2.py.
"""
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(_REPO)
sys.path[:0] = [str(ROOT / "environments/real/CTCBench")]
from ctcbench_real import env as E  # noqa: E402

QD_RUNGS = ("r2k", "r4k", "r8k", "r16k", "r32k", "r64k", "r128k", "r256k")
HP_RUNGS = ("r2k", "r4k", "r8k", "r16k", "r32k", "r64k", "r128k", "r256k", "r512k", "r1m")
N = 13


def toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())


def norm(s: str) -> str:
    return " ".join(toks(s))


def text(item) -> str:
    return item["text"] if isinstance(item, dict) else str(item)


def grams(s: str) -> set:
    t = toks(s)
    return {tuple(t[i:i + N]) for i in range(len(t) - N + 1)}


def banned() -> tuple[set, set, set, dict]:
    bq, bd, counts = set(), set(), {}
    for r in QD_RUNGS:
        n = 0
        for ex in E._hf_rung("qdmatch_hpqa", r):
            docs = ex["documents"]
            for q, d in ex["gold_pairs"]:
                bq.add(norm(text(docs[q - 1])))
                bd.add(norm(text(docs[d - 1])))
            n += 1
        counts[f"qdmatch_hpqa/{r}"] = n
    for r in HP_RUNGS:
        n = 0
        for ex in E._hf_rung("hotpotqa", r):
            for q in ex.get("queries") or []:
                bq.add(norm(q))
            docs = ex["documents"]
            for i in ex.get("gold_doc_indices") or []:
                bd.add(norm(text(docs[int(i)])))
            n += 1
        counts[f"hotpotqa/{r}"] = n
    bq.discard("")
    bd.discard("")
    bg = set().union(*(grams(q) for q in bq))
    return bq, bd, bg, counts


def check(ex: dict, bq: set, bd: set, bg: set) -> list[str]:
    why = []
    for k, it in enumerate(ex["documents"]):
        t = norm(text(it))
        is_q = isinstance(it, dict) and it.get("type") == "query"
        if t in bq:
            why.append(f"item{k + 1}:{'query' if is_q else 'doc'}=eval_relevant_query")
        if t in bd:
            why.append(f"item{k + 1}:{'query' if is_q else 'doc'}=eval_gold_doc")
        if is_q and t not in bq and grams(t) & bg:
            why.append(f"item{k + 1}:query~13gram_eval_query")
    return why


def main(out: str, files: list[str]) -> None:
    bq, bd, bg, counts = banned()
    rep = {"rule": __doc__.split("\n\n")[1], "eval_rungs_rows": counts, "banned_queries": len(bq),
           "banned_gold_docs": len(bd), "files": {}}
    for f in files:
        bad = []
        n = 0
        for k, line in enumerate(open(f)):
            ex = json.loads(line)
            n += 1
            why = check(ex, bq, bd, bg)
            if why:
                bad.append({"pos": k, "gen_row": ex.get("_gen_row"), "n_hits": len(why), "reasons": why[:8]})
        rep["files"][f] = {"rows": n, "contaminated": len(bad), "clean": n - len(bad), "bad": bad}
        print(f, "rows", n, "contaminated", len(bad), flush=True)
    Path(out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
