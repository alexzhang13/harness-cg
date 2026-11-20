"""Trajectory statistics of the held-out suite (Table main-composition-stats; per-benchmark tables of App. C).

    python stats.py <OUT> [--usage-dir <LB_USAGE_DIR> --usage-tag <ACCT_TAG>] [aggregate.py selection flags] [--json f]

Same trajectories as aggregate.py. Per trajectory: turns, sub-calls, delegated, submitted, cap (hit the turn
limit), loop (same assistant turn 3+ times in a row) and tokens. Tokens are measured usage where recorded (or from
the router with --usage-dir), else characters / 4.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate as A  # noqa: E402

METRICS = ["reward", "turns", "sub_calls", "delegated", "submitted", "cap", "loop",
           "root_in", "root_out", "sub_in", "sub_out", "root_estimated", "sub_estimated"]


def val(r, key, default=None):
    x = r.get(key)
    return x if x is not None else (r.get("meta") or {}).get(key, default)


def messages(r):
    """The message list with the most assistant turns."""
    candidates = []
    for key in ("messages", "completion", "history", "turns"):
        v = r.get(key) or (r.get("meta") or {}).get(key)
        if isinstance(v, list) and v:
            count = sum(isinstance(x, dict) and x.get("role") == "assistant" for x in v)
            if count:
                candidates.append((count, v))
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


def token_features(r, trace, ms, sub):
    result = dict(root_in=None, root_out=None, sub_in=None, sub_out=None, root_estimated=0., sub_estimated=0.)

    def usage(calls, expected):
        if not calls or len(calls) != expected:
            return None
        us = [c.get("usage") for c in calls]
        if all(isinstance(u, dict) and u.get("prompt_tokens") is not None and
               u.get("completion_tokens") is not None for u in us):
            return sum(u["prompt_tokens"] for u in us), sum(u["completion_tokens"] for u in us)
    root_usage = None
    tu = r.get("token_usage") or {}
    if tu.get("input_tokens") is not None and tu.get("output_tokens") is not None:
        root_usage = tu["input_tokens"], tu["output_tokens"]
    if root_usage is None:
        calls = r.get("call_meter") or []
        root_usage = usage([c for c in calls if c.get("root")], r.get("iterations"))
    if root_usage is None and trace:
        calls = [x.get("response") or {} for x in trace.get("trajectory") or []]
        root_usage = usage(calls, len([x for x in (ms or []) if x.get("role") == "assistant"]))
    if root_usage:
        result.update(root_in=root_usage[0], root_out=root_usage[1])
    else:
        seq = r.get("messages") or ((r.get("prompt") or []) + (r.get("completion") or []))
        if seq:
            ri = ro = prefix = 0
            for item in seq:
                if not isinstance(item, dict):
                    continue
                content = item.get("content") or ""
                n = len(content) if isinstance(content, str) else len(json.dumps(content))
                reasoning = item.get("reasoning_content") or item.get("reasoning") or ""
                n += len(reasoning) if isinstance(reasoning, str) else 0
                if item.get("role") == "assistant":
                    ri += prefix
                    ro += n
                prefix += n
            result.update(root_in=ri / 4, root_out=ro / 4, root_estimated=1.)
    if sub == 0:
        result.update(sub_in=0., sub_out=0.)
        return result
    measured = usage([c for c in r.get("call_meter") or [] if not c.get("root")], sub)
    if measured:
        result.update(sub_in=measured[0], sub_out=measured[1])
        return result
    pi, po = val(r, "sub_prompt_chars"), val(r, "sub_response_chars")
    children = r.get("subcalls") or r.get("sub_log")
    if children is None and trace:
        children = trace.get("sub_calls")
    if isinstance(children, list) and len(children) == sub:
        if pi is None and all(c.get("prompt_chars") is not None for c in children):
            pi = sum(c["prompt_chars"] for c in children)
        if po is None and all(c.get("response_chars") is not None or
                              isinstance(c.get("response"), str) for c in children):
            po = sum(c.get("response_chars", len(c.get("response") or "")) for c in children)
    if trace and (pi is None or po is None):
        calls = [c for c in trace.get("calls") or [] if c.get("root") is False]
        if len(calls) == sub:
            if pi is None and all(isinstance(c.get("prompt"), list) for c in calls):
                pi = sum(len(m.get("content") or "") for c in calls for m in c["prompt"])
            if po is None and all(isinstance(c.get("response"), dict) and
                                  isinstance(c["response"].get("content"), str) for c in calls):
                po = sum(len(c["response"]["content"]) for c in calls)
    result.update(sub_in=pi / 4 if pi is not None else None,
                  sub_out=po / 4 if po is not None else None, sub_estimated=1.)
    return result


def compact(bench, row, dead_as_zero):
    r = row.rec
    trace = None
    if row.trace and Path(row.trace).exists():
        trace = json.loads(Path(row.trace).read_text())
    m = r.get("metrics") or {}
    turns = r.get("iterations", m.get("rlm_iterations", val(r, "rlm_iterations")))
    sub = r.get("sub_calls", m.get("rlm_sub_llm_calls", val(r, "rlm_sub_llm_calls")))
    stop = val(r, "stop_condition", "") or ""
    ms = messages(r) or (messages(trace) if trace else None)
    if turns is None and ms:
        turns = sum(isinstance(x, dict) and x.get("role") == "assistant" for x in ms)
    if sub is None and trace:
        sub = len(trace.get("sub_calls") or [])
    submitted = m.get("rlm_has_final_answer")
    if submitted is None:
        submitted = stop == "has_final_answer" or val(r, "final_answer", r.get("final")) is not None
    if dead_as_zero and stop == "repl_worker_dead":
        submitted = 0
    text = [str(x.get("content") or "") for x in (ms or []) if isinstance(x, dict) and x.get("role") == "assistant"]
    norm = [re.sub(r"\s+", " ", t).strip() for t in text]
    run = longest = 1 if norm else 0
    for a, b in zip(norm, norm[1:]):
        run = run + 1 if b and a == b else 1
        longest = max(longest, run)
    cap = A.TURN_CAP[bench]
    out = dict(reward=row.reward, turns=float(turns) if turns is not None else None,
               sub_calls=float(sub) if sub is not None else None,
               delegated=float(sub > 0) if sub is not None else None, submitted=float(bool(submitted)),
               cap=float(turns >= cap) if turns is not None else None, loop=float(longest >= 3) if ms else None)
    out.update(token_features(r, trace, ms, sub))
    return out


def usage_totals(usage_dir: Path, tags: list[str]) -> dict:
    tot = {"root_in": 0, "root_out": 0, "sub_in": 0, "sub_out": 0, "requests": 0}
    for t in tags:
        for rec in A._lines(usage_dir / f"{t}.jsonl"):
            if rec.get("status") != 200 or rec.get("pt") is None:
                continue
            side = "root" if rec.get("role") == "root" else "sub"
            tot[f"{side}_in"] += rec["pt"]
            tot[f"{side}_out"] += rec.get("ct") or 0
            tot["requests"] += 1
    return tot


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outs", nargs="+", type=Path)
    ap.add_argument("--usage-dir", type=Path, default=None)
    ap.add_argument("--usage-tag", default=None, help="the ACCT_TAG prefix the run used")
    ap.add_argument("--bcplus", choices=["strict", "judged", "env"], default="strict")
    ap.add_argument("--longreason", choices=["subthink", "nothink"], default=None)
    ap.add_argument("--oolong", choices=["nothink", "low"], default=None)
    ap.add_argument("--dead-as-zero", action="store_true")
    ap.add_argument("--common", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    if a.usage_dir and len(a.outs) > 1:
        raise SystemExit("--usage-dir reads one run's tags: pass one OUT")
    runs = [A.select_all(o, a) for o in a.outs]
    if a.common and len(runs) > 1:
        A.restrict_common(runs)
    report = {}
    for out, sel in zip(a.outs, runs):
        per = {}
        for name, s in sel.items():
            bench = name.split(":")[0]
            rows = [compact(bench, r, a.dead_as_zero) for r in s.valid.values()]
            if not rows:
                continue
            per[name] = {"n": len(rows), **{k: mean_or_none([x[k] for x in rows]) for k in METRICS},
                         "coverage": {k: sum(x[k] is not None for x in rows) for k in METRICS}}
            if a.usage_dir:
                if bench == "ctc":
                    tags = [f"{a.usage_tag}-ctc-{name[4:].replace('@', '_')}"]
                elif bench == "elb":
                    tags = [f"{a.usage_tag}-elb-{t[:3]}" for t in A.ELB_TASKS]
                else:
                    tags = [f"{a.usage_tag}-{bench}"]
                tot = usage_totals(a.usage_dir, tags)
                per[name].update({k: tot[k] / len(rows) for k in ("root_in", "root_out", "sub_in", "sub_out")},
                                 root_estimated=0., sub_estimated=0., proxy_requests=tot["requests"])
        bench_rows = {b: v for b, v in per.items() if not b.startswith("ctc:")}
        cells = [per.get("ctc:" + c) for c in A.CTC_CELLS]
        if all(cells):
            bench_rows["ctc"] = {"n": sum(c["n"] for c in cells),
                                 **{k: mean_or_none([c[k] for c in cells]) if all(c[k] is not None for c in cells)
                                    else None for k in METRICS}}
        macro = ({k: mean_or_none([bench_rows[b][k] for b in A.BENCHMARKS])
                  if all(bench_rows[b][k] is not None for b in A.BENCHMARKS) else None for k in METRICS}
                 if all(b in bench_rows for b in A.BENCHMARKS) else None)
        print(f"\n== {out}")
        hdr = f"{'benchmark':26s} {'n':>5s} {'score':>6s} {'turns':>6s} {'subs':>8s} {'deleg%':>7s} {'cap%':>6s} " \
              f"{'loop%':>6s} {'subm%':>6s} {'root in/out (K)':>16s} {'sub in/out (K)':>16s}"
        print(hdr)

        def line(label, v):
            def f(x, scale=1.0, w=6, d=1):
                return f"{x * scale:{w}.{d}f}" if x is not None else f"{'-':>{w}s}"
            print(f"{label:26s} {v.get('n', 0):5d} {f(v['reward'], 100)} {f(v['turns'])} {f(v['sub_calls'], w=8)} "
                  f"{f(v['delegated'], 100, 7)} {f(v['cap'], 100)} {f(v['loop'], 100)} {f(v['submitted'], 100)} "
                  f"{f(v['root_in'], 1e-3, 7, 0)} / {f(v['root_out'], 1e-3, 6, 0)} "
                  f"{f(v['sub_in'], 1e-3, 7, 0)} / {f(v['sub_out'], 1e-3, 6, 0)}")
        for b in A.BENCHMARKS:
            if b == "ctc":
                for c in A.CTC_CELLS:
                    if per.get("ctc:" + c):
                        line("  " + c, per["ctc:" + c])
            if b in bench_rows:
                line(A.LABELS[b], bench_rows[b])
        if macro:
            line("Avg (macro over 8)", {**macro, "n": sum(bench_rows[b]["n"] for b in A.BENCHMARKS)})
        report[str(out)] = {"per_task": per, "benchmarks": bench_rows, "macro": macro,
                            "token_basis": "router usage per reported item" if a.usage_dir else "per-trajectory"}
    if a.json:
        a.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
