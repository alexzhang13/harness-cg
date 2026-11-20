"""Held-out suite scores from the directories written by run_suite.sh.

    python aggregate.py <OUT> [<OUT> ...] [--bcplus strict|judged|env] [--common] [--dead-as-zero] [--json f]
    python aggregate.py <OUT> --invalidate        # mark invalid items for a rerun of run_suite.sh

Per item the first valid attempt counts. Attempts that failed for infrastructure reasons (dead REPL worker, timeout,
request error, failed OBLIQ sub-call, failed ELB judge) are invalid; items with only invalid attempts are left out and
listed. Hitting the turn cap without an answer is a valid 0. CTC is the mean of its five cells; Avg is the mean of
the eight benchmarks. --common keeps only items valid in every given run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

BENCHMARKS = ["longreason", "loogle", "ctc", "bcplus", "oolong", "corpusqa", "obliq", "elb"]
LABELS = {"longreason": "LongReason", "loogle": "LooGLE", "ctc": "CTC", "bcplus": "BC+", "oolong": "OOLONG-Real",
          "corpusqa": "CorpusQA", "obliq": "OBLIQ", "elb": "ELB"}
CTC_CELLS = ["qdmatch_hpqa@r256k", "contradiction_iid@r1m", "outlier@r1m", "xabsence_paraphrase@r32k", "grouping@r32k"]
EXPECTED = {"longreason": 794, "loogle": 206, "bcplus": 830, "oolong": 577, "corpusqa": 81, "obliq": 151, "elb": 350,
            "qdmatch_hpqa@r256k": 125, "contradiction_iid@r1m": 125, "outlier@r1m": 125,
            "xabsence_paraphrase@r32k": 100, "grouping@r32k": 60}
ELB_TASKS = {"T01LegalMDS": 100, "T03MaterialSEG": 50, "T04EduPAE": 100, "T06HealthCNG": 100}
TURN_CAP = {"longreason": 40, "loogle": 12, "ctc": 30, "bcplus": 40, "oolong": 40, "corpusqa": 30, "obliq": 20,
            "elb": 40}
OBLIQ_INFRA = ("Proxy request failed", "exceeded parent watchdog", "exceeded 600s timeout")


@dataclass
class Row:
    key: object
    reward: float
    rec: dict
    path: str
    trace: str | None = None


@dataclass
class Selection:
    valid: dict = field(default_factory=dict)       # key -> Row
    invalid: dict = field(default_factory=dict)     # key -> (reason, last record, path)
    expected: int = 0


def _lines(path):
    try:
        with open(path) as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def pick_dir(out: Path, base: str, variants: list[str], choice: str | None) -> Path | None:
    """The <base>_<variant> directory (the requested variant, or the only one)."""
    if choice:
        return out / f"{base}_{choice}"
    have = [out / f"{base}_{v}" for v in variants if (out / f"{base}_{v}").exists()]
    if len(have) > 1:
        raise SystemExit(f"{out}: several {base} variants ({', '.join(p.name for p in have)}); pass --{base}")
    return have[0] if have else None


def _attempt_invalid(rec: dict, bench: str, dead_as_zero: bool) -> str | None:
    stop = str(rec.get("stop_condition") or "")
    if stop == "repl_worker_dead":
        return None if dead_as_zero else "repl_worker_dead"
    if rec.get("error"):
        return "error: " + str(rec.get("error"))[:120]
    if "timeout" in stop:
        return stop
    if rec.get("reward") is None:
        return "no reward"
    if bench == "obliq":
        txt = json.dumps(rec.get("completion") or [])
        hit = next((m for m in OBLIQ_INFRA if m in txt), None)
        if hit:
            return f"failed sub-call ({hit})"
    return None


def _reward(rec: dict) -> float:
    if rec.get("stop_condition") == "repl_worker_dead":
        return 0.0
    return float(rec.get("reward") or 0.0)


def select_run_eval(cache_dir: Path, bench: str, dead_as_zero: bool) -> Selection:
    sel = Selection()
    recs = []
    for p in sorted(glob.glob(str(cache_dir / "*.jsonl"))):
        recs += [(str(r.get("ts") or ""), r, p) for r in _lines(p) if r.get("key") is not None]
    recs.sort(key=lambda x: x[0])
    for _, r, p in recs:
        k = r["key"]
        if k in sel.valid:
            continue
        why = _attempt_invalid(r, bench, dead_as_zero)
        if why is None:
            sel.valid[k] = Row(k, _reward(r), r, p)
            sel.invalid.pop(k, None)
        else:
            sel.invalid[k] = (why, r, p)
    return sel


def select_oolong(d: Path, dead_as_zero: bool) -> Selection:
    sel = Selection()
    p = d / "rows.jsonl"
    for r in _lines(p):
        k = r["id"]
        if k in sel.valid:
            continue
        why = _attempt_invalid(r, "oolong", dead_as_zero)
        if why is None:
            sel.valid[k] = Row(k, _reward(r), r, str(p), str(d / "traces" / f"{k}.json"))
            sel.invalid.pop(k, None)
        else:
            sel.invalid[k] = (why, r, str(p))
    return sel


def select_loogle(d: Path) -> Selection:
    sel = Selection()
    p = d / "rows.jsonl"
    for r in _lines(p):
        k = r["idx"]
        if k in sel.valid:
            continue
        if r.get("iterations") is None:
            sel.invalid[k] = ("no iterations: " + str(r.get("error"))[:120], r, str(p))
        else:
            sel.valid[k] = Row(k, float(r.get("reward") or 0.0), r, str(p), str(d / f"trace_{k}.json"))
            sel.invalid.pop(k, None)
    return sel


def elb_files(d: Path) -> list[str]:
    files = sorted(f for f in glob.glob(str(d / "*.jsonl")) if not f.endswith(".rejudged.jsonl"))
    files.sort(key=lambda f: ("-redo" in os.path.basename(f), f))
    return [f[:-6] + ".rejudged.jsonl" if os.path.exists(f[:-6] + ".rejudged.jsonl") else f for f in files]


def select_elb(d: Path, dead_as_zero: bool) -> Selection:
    sel = Selection()
    for p in elb_files(d):
        for r in _lines(p):
            k = (r["task"], r["example_id"])
            if k in sel.valid:
                continue
            why = _attempt_invalid(r, "elb", dead_as_zero)
            if why is None and r.get("judge_error") and r.get("stop_condition") != "repl_worker_dead":
                why = "judge_error (run elb_rejudge.py)"
            if why is None:
                sel.valid[k] = Row(k, _reward(r), r, p)
                sel.invalid.pop(k, None)
            else:
                sel.invalid[k] = (why, r, p)
    return sel


def strict_bcplus(reward: float, final, gold) -> float:
    """Zero a credit whose gold parts only occur inside other words."""
    import browsecomp_plus_real.env as E
    if not reward or not final or not gold:
        return reward or 0.0
    pred = " " + E._normalize_answer(str(final)) + " "
    parts = E._split_gold(str(gold))
    sub = all(E._part_in_pred(p, pred) for p in parts)
    word = all(re.search(r"(?<![0-9a-z])" + re.escape(p) + r"(?![0-9a-z])", pred) for p in parts)
    return 0.0 if (sub and not word) else reward


def rescore_bcplus(sel: Selection, mode: str, out: Path) -> None:
    if mode == "env":
        return
    cache = {}
    if mode == "judged":
        p = out / "bcplus" / "judge_cache.json"
        if not p.exists():
            raise SystemExit(f"{p} missing: run bcplus_judge_regrade.py first (or pass --bcplus strict)")
        cache = json.loads(p.read_text())
    for row in sel.valid.values():
        r = row.rec
        if mode == "strict":
            row.reward = strict_bcplus(row.reward, r.get("final_answer"), r.get("answer"))
        elif row.reward < 1 and str(cache.get(str(row.key), "")).startswith("yes"):
            row.reward = 1.0


def select_all(out: Path, a) -> dict[str, Selection]:
    """{benchmark or 'ctc:<cell>': Selection} for the benchmarks under `out`."""
    res = {}
    lr = pick_dir(out, "longreason", ["subthink", "nothink"], getattr(a, "longreason", None))
    if lr and lr.exists():
        res["longreason"] = select_run_eval(lr / "cache", "longreason", a.dead_as_zero)
    if (out / "loogle").exists():
        res["loogle"] = select_loogle(out / "loogle")
    for cell in CTC_CELLS:
        if (out / "ctc" / cell).exists():
            res["ctc:" + cell] = select_run_eval(out / "ctc" / cell / "cache", "ctc", a.dead_as_zero)
            res["ctc:" + cell].expected = EXPECTED[cell]
    for b in ("bcplus", "corpusqa", "obliq"):
        if (out / b).exists():
            res[b] = select_run_eval(out / b / "cache", b, a.dead_as_zero)
    if "bcplus" in res:
        rescore_bcplus(res["bcplus"], a.bcplus, out)
    oo = pick_dir(out, "oolong", ["nothink", "low"], getattr(a, "oolong", None))
    if oo and oo.exists():
        res["oolong"] = select_oolong(oo, a.dead_as_zero)
    if (out / "elb").exists():
        res["elb"] = select_elb(out / "elb", a.dead_as_zero)
    for b, s in res.items():
        s.expected = s.expected or EXPECTED.get(b, 0)
    return res


def restrict_common(runs: list[dict[str, Selection]]) -> None:
    for b in set.intersection(*[set(r) for r in runs]):
        keep = set.intersection(*[set(r[b].valid) for r in runs])
        for r in runs:
            r[b].valid = {k: v for k, v in r[b].valid.items() if k in keep}


def scores(sel: dict[str, Selection]) -> dict[str, float | None]:
    def mean(s):
        return statistics.mean(r.reward for r in s.valid.values()) if s and s.valid else None
    out = {b: mean(sel.get(b)) for b in BENCHMARKS if b != "ctc"}
    cells = [mean(sel.get("ctc:" + c)) for c in CTC_CELLS]
    out["ctc"] = statistics.mean(cells) if all(c is not None for c in cells) else None
    for c, v in zip(CTC_CELLS, cells):
        out["ctc:" + c] = v
    main = [out[b] for b in BENCHMARKS]
    out["avg"] = statistics.mean(main) if all(v is not None for v in main) else None
    return out


def invalidate(out: Path, sel: dict[str, Selection]) -> None:
    """Mark items with only invalid attempts for a rerun."""
    now = datetime.now(timezone.utc).isoformat()
    for b, s in sel.items():
        if not s.invalid:
            continue
        if b == "loogle":
            p = out / "loogle" / "rows.jsonl"
            bad = {k for k in s.invalid}
            rows = list(_lines(p))
            with open(out / "loogle" / "rows.invalid.jsonl", "a") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows if r["idx"] in bad)
            p.write_text("".join(json.dumps(r) + "\n" for r in rows if r["idx"] not in bad))
        elif b == "oolong":
            p = Path(next(iter(s.invalid.values()))[2])
            with open(p, "a") as f:
                for k, (why, r, _) in s.invalid.items():
                    if not r.get("error"):
                        f.write(json.dumps({**r, "reward": 0.0, "error": f"invalidated: {why}"}) + "\n")
        elif b == "elb":
            jobs = out / "elb" / "redo.jobs"
            n = sum(1 for _ in open(jobs)) if jobs.exists() else 0
            by_task: dict = {}
            for (task, eid), (why, _, _) in s.invalid.items():
                if not why.startswith("judge_error"):
                    by_task.setdefault(task, []).append(eid)
            with open(jobs, "a") as f:
                for task, ids in sorted(by_task.items()):
                    n += 1
                    f.write(f"redo{n}-{task[:3]} {task} {','.join(map(str, sorted(ids)))}\n")
        else:
            for k, (why, r, p) in s.invalid.items():
                if r.get("error"):
                    continue
                m = re.match(r"^(.*__[0-9a-f]{12})\.", os.path.basename(p))
                if not m:
                    continue
                with open(os.path.join(os.path.dirname(p), m.group(1) + ".invalidated.jsonl"), "a") as f:
                    f.write(json.dumps({"key": k, "example_id": r.get("example_id"), "reward": None,
                                        "error": f"invalidated: {why}", "ts": now}) + "\n")
        print(f"{b}: marked {len(s.invalid)} invalid item(s) for rerun")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outs", nargs="+", type=Path)
    ap.add_argument("--bcplus", choices=["strict", "judged", "env"], default="strict")
    ap.add_argument("--longreason", choices=["subthink", "nothink"], default=None)
    ap.add_argument("--oolong", choices=["nothink", "low"], default=None)
    ap.add_argument("--dead-as-zero", action="store_true")
    ap.add_argument("--common", action="store_true")
    ap.add_argument("--invalidate", action="store_true")
    ap.add_argument("--show-invalid", type=int, default=10, help="invalid items listed per benchmark")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    runs = [select_all(o, a) for o in a.outs]
    if a.common and len(runs) > 1:
        restrict_common(runs)
    report = {}
    for out, sel in zip(a.outs, runs):
        sc = scores(sel)
        print(f"\n== {out}")
        print(f"{'benchmark':28s} {'score':>7s} {'valid':>6s} {'invalid':>8s} {'expected':>9s}")
        for b in BENCHMARKS:
            parts = ["ctc:" + c for c in CTC_CELLS] if b == "ctc" else [b]
            for name in parts:
                s = sel.get(name)
                v = sc.get(name)
                label = ("  " + name[4:]) if name.startswith("ctc:") else LABELS[b]
                print(f"{label:28s} {100 * v if v is not None else float('nan'):7.1f} "
                      f"{len(s.valid) if s else 0:6d} {len(s.invalid) if s else 0:8d} {s.expected if s else EXPECTED.get(name, 0):9d}")
            if b == "ctc":
                print(f"{'CTC (mean of 5 cells)':28s} {100 * sc['ctc'] if sc['ctc'] is not None else float('nan'):7.1f}")
        print(f"{'Avg':28s} {100 * sc['avg'] if sc['avg'] is not None else float('nan'):7.1f}")
        for name, s in sel.items():
            if s.invalid:
                items = list(s.invalid.items())[:a.show_invalid]
                print(f"  invalid {name}: " + "; ".join(f"{k}: {why}" for k, (why, _, _) in items)
                      + (" ..." if len(s.invalid) > a.show_invalid else ""))
        report[str(out)] = {"scores": sc, "bcplus_mode": a.bcplus, "dead_as_zero": a.dead_as_zero,
                            "n_valid": {b: len(s.valid) for b, s in sel.items()},
                            "invalid": {b: {str(k): why for k, (why, _, _) in s.invalid.items()} for b, s in sel.items()}}
        if a.invalidate:
            invalidate(out, sel)
    if a.json:
        a.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
