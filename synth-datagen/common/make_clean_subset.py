"""Cut the clean set: keep rows with ``synthetic`` and ``clean`` true.

    python make_clean_subset.py mark <builder rows.jsonl> <full set config.jsonl> <out.jsonl>
    python make_clean_subset.py filter <full set dir> <out dir>

``mark`` copies the ``clean`` / ``eval_overlap`` columns from the published full set into a builder's output.
"""
import argparse
import json
from pathlib import Path


def _key(r: dict) -> tuple:
    return (r.get("source_store"), r.get("source_uid"), r.get("descriptor"))


def mark(rows_path: str, full_path: str, out_path: str) -> None:
    verdict = {}
    for line in open(full_path):
        r = json.loads(line)
        verdict[_key(r)] = (r.get("clean", True), r.get("eval_overlap", ""))
    n_bad = 0
    with open(out_path, "w") as f:
        for line in open(rows_path):
            r = json.loads(line)
            clean, overlap = verdict.get(_key(r), (True, ""))
            n_bad += not clean
            out = {}
            for k, v in r.items():
                if k in ("clean", "eval_overlap"):
                    continue
                out[k] = v
                if k == "synthetic":
                    out["clean"], out["eval_overlap"] = clean, overlap
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"{out_path}: {n_bad} rows marked not clean")


def filter_clean(full_dir: str, out_dir: str) -> None:
    (Path(out_dir) / "data").mkdir(parents=True, exist_ok=True)
    for src in sorted((Path(full_dir) / "data").glob("*.jsonl")):
        kept = 0
        with open(Path(out_dir) / "data" / src.name, "w") as f:
            for line in open(src):
                r = json.loads(line)
                if r.get("synthetic") is True and r.get("clean") is True:
                    f.write(line if line.endswith("\n") else line + "\n")
                    kept += 1
        print(f"{src.stem}: {kept} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mark")
    m.add_argument("rows")
    m.add_argument("full")
    m.add_argument("out")
    f = sub.add_parser("filter")
    f.add_argument("full_dir")
    f.add_argument("out_dir")
    a = ap.parse_args()
    if a.cmd == "mark":
        mark(a.rows, a.full, a.out)
    else:
        filter_clean(a.full_dir, a.out_dir)


if __name__ == "__main__":
    main()
