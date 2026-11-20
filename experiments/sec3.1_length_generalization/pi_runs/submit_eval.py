"""Keep up to --servers-per-run eval servers (2 GPUs each) per unfinished Pi run, at most --max-jobs in total.

    python submit_eval.py --dry-run
    python submit_eval.py --tasks mrcr --seeds 0 --servers-per-run 2 --max-jobs 6
"""

from __future__ import annotations

import argparse
import getpass
import subprocess
from collections import Counter

import eval_tools as E
import settings as S

SBATCH = S.HERE / "slurm" / "eval_server.sbatch"
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}


def eval_job_name(task: str, seed: int) -> str:
    return f"cgpi-eval-{task}-s{seed}{S.JOB_TAG}"


def queued_eval_jobs() -> Counter:
    out = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%j %T"], capture_output=True, text=True).stdout
    c: Counter = Counter()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].startswith("cgpi-eval-") and parts[1] in ACTIVE:
            c[parts[0]] += 1
    return c


def plan(selection, tasks_n: dict[str, int], alive: Counter, servers_per_run: int, max_jobs: int, unit_size: int = E.UNIT_SIZE):
    """[(task, seed, state, alive, to_submit)]."""
    total = sum(alive.values())
    rows = []
    for task, seed in selection:
        run = S.run_dir(task, seed)
        shards = E.n_shards(task, tasks_n, unit_size)
        st = E.eval_state(run, shards)
        name = eval_job_name(task, seed)
        have = alive.get(name, 0)
        want = min(servers_per_run, st["remaining_units"])
        n = max(0, min(want - have, max_jobs - total))
        total += n
        rows.append((task, seed, st, have, n, want - have > n and want - have > 0))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tasks_n = E.driver_tasks()
    ap.add_argument("--tasks", default=",".join(t for t in S.TASKS if t in tasks_n), help="subset of: " + ",".join(tasks_n))
    ap.add_argument("--seeds", default=",".join(map(str, S.SEEDS)))
    ap.add_argument("--servers-per-run", type=int, default=2)
    ap.add_argument("--max-jobs", type=int, default=6, help="cap on this tool's eval jobs pending or running")
    ap.add_argument("--partition", default=S.SLURM["partition"])
    ap.add_argument("--time", default="1-00:00:00")
    ap.add_argument("--exit-after-idle", type=int, default=3600)
    ap.add_argument("--sbatch-args", default="", help="extra sbatch arguments, e.g. '--nodelist=<node>'")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks, seeds = args.tasks.split(","), [int(s) for s in args.seeds.split(",")]
    bad = [t for t in tasks if t not in tasks_n]
    if bad:
        raise SystemExit(f"tasks without an eval driver spec: {bad} (supported: {sorted(tasks_n)})")
    unit = E.driver_unit_size()
    rows = plan(S.matrix(tasks, seeds), tasks_n, queued_eval_jobs(), args.servers_per_run, args.max_jobs, unit)

    print(f"{'task':13s} {'seed':>6s} {'adapters':>8s} {'evaluated':>9s} {'units left':>10s} {'alive':>5s} action")
    submitted = 0
    for task, seed, st, have, n, deferred in rows:
        run, name = S.run_dir(task, seed), eval_job_name(task, seed)
        if E.fully_evaluated(run, E.n_shards(task, tasks_n, unit)):
            action = "done"
        elif st["remaining_units"] == 0:
            action = "waiting for adapters"
        elif n == 0:
            action = "deferred (max-jobs)" if deferred else "ok"
        else:
            cmd = ["sbatch", "-J", name, *(["-p", args.partition] if args.partition else []), "-t", args.time, "--gres=gpu:2",
                   "-o", str(run / "eval" / "logs" / "slurm-%j.out"), *args.sbatch_args.split(),
                   f"--export=ALL,PI_RUNS_DIR={S.HERE},RUN={run},TASK={task},EXIT_AFTER_IDLE={args.exit_after_idle}", str(SBATCH)]
            if args.dry_run:
                action = f"would submit {n}: " + " ".join(cmd)
            else:
                (run / "eval" / "logs").mkdir(parents=True, exist_ok=True)
                msgs = []
                for _ in range(n):
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    msgs.append(r.stdout.strip() or r.stderr.strip())
                    submitted += r.returncode == 0
                action = f"submitted {n}: " + "; ".join(msgs)
        print(f"{task:13s} {seed:>6d} {len(st['adapters']):>5d}/{len(S.EXPECTED_STEPS):<2d} {len(st['evaluated']):>9d} "
              f"{st['remaining_units']:>10d} {have:>5d} {action}")
    print(f"{submitted} submitted")


if __name__ == "__main__":
    main()
