"""Evaluation status of the Pi runs: adapters, evaluated checkpoints, mean reward, errors, cap hits, servers.

    python eval_status.py                      # one row per run
    python eval_status.py --curves             # plus one line per run: step:mean reward
    python eval_status.py --tasks mrcr --seeds 0
"""

from __future__ import annotations

import argparse

import eval_tools as E
import settings as S
import submit_eval as SE


def main() -> None:
    tasks_n = E.driver_tasks()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(t for t in S.TASKS if t in tasks_n))
    ap.add_argument("--seeds", default=",".join(map(str, S.SEEDS)))
    ap.add_argument("--curves", action="store_true")
    args = ap.parse_args()
    unit = E.driver_unit_size()
    alive = SE.queued_eval_jobs()
    print(f"{'task':13s} {'seed':>6s} {'adapters':>8s} {'evaluated':>9s} {'partial':>7s} {'servers':>7s} {'errors':>6s} {'cap-hit':>7s}  mean reward (latest evaluated)")
    for task, seed in S.matrix(args.tasks.split(","), [int(s) for s in args.seeds.split(",")]):
        run = S.run_dir(task, seed)
        shards = E.n_shards(task, tasks_n, unit)
        st = E.eval_state(run, shards)
        stats = {s: E.step_stats(run, s) for s in st["evaluated"]}
        errs = sum(v["errors"] for v in stats.values())
        caps = sum(v["cap_hits"] for v in stats.values())
        partial = sum(1 for s, m in st["pending"].items() if m < shards)
        last = max(stats, default=None)
        latest = f"step_{last}: {stats[last]['mean']:.3f} (n={stats[last]['n']})" if last is not None else "-"
        print(f"{task:13s} {seed:>6d} {len(st['adapters']):>5d}/{len(S.EXPECTED_STEPS):<2d} {len(st['evaluated']):>9d} "
              f"{partial:>7d} {alive.get(SE.eval_job_name(task, seed), 0):>7d} {errs:>6d} {caps:>7d}  {latest}")
        if args.curves and stats:
            print("    " + "  ".join(f"{s}:{v['mean']:.3f}" for s, v in sorted(stats.items())))


if __name__ == "__main__":
    main()
