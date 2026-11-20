"""Table recursion-composition and its statistics table from run_gated.py output directories.

    python stats.py runs/gated/<model>_<cond> [...]

Partial solve = mean task score per chain (unreached tasks 0); full solve = share of chains with every task passed.
"""
import collections, glob, json, statistics as st, sys


def rlm_call_stats(levels):
    turns, subs = [], []
    for lv in levels:
        traj = lv.get("traj") or {}
        its = traj.get("iterations") or []
        if not its:
            continue
        turns.append(len(its))
        subs.append(sum(len((cb.get("result") or {}).get("rlm_calls") or []) for it in its for cb in it.get("code_blocks", [])))
    return turns, subs


for d in sys.argv[1:]:
    rows = {}
    for f in sorted(glob.glob(d + "/gated_*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line); rows[(r["cond"], r["tag"])] = r
    R = list(rows.values())
    if not R:
        print(d, "no runs"); continue
    n = len(R)
    passed = [r.get("depth", 0) for r in R]
    tc = lambda k: [(r.get("tool_calls") or {}).get(k, 0) for r in R]
    calls = [len(r.get("levels") or []) or 1 for r in R]
    att = [a for r in R for a in (r.get("attempt_pass") or []) if a]
    turns, subs = [], []
    for r in R:
        t, s = rlm_call_stats(r.get("levels") or []); turns += t; subs += s
    out = {
        "chains": n,
        "partial_solve_pct": round(100 * st.mean(r["mean"] for r in R), 1),
        "full_solve_pct": round(100 * sum(bool(r["all"]) for r in R) / n, 1),
        "chains_by_tasks_passed": dict(sorted(collections.Counter(passed).items())),
        "per_chain": {"passed": round(st.mean(passed), 2), "unlock": round(st.mean(tc("unlock")), 2),
                      "handoff": round(st.mean(tc("handoff")), 2), "rlm_calls": round(st.mean(calls), 2)},
        "per_rlm_call": {"turns": round(st.mean(turns), 1) if turns else None, "sub_calls": round(st.mean(subs), 2) if subs else None},
        "second_attempt_pct": round(100 * sum(a == 2 for a in att) / len(att), 1) if att else None,
        "errors": sum(bool(r.get("error")) for r in R),
    }
    print(d, json.dumps(out))
