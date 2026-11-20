"""Submit missing Pi runs to SLURM, one 8-GPU node each; complete, queued and running ones are skipped.

    python submit.py [--tasks mrcr,adaleval]
    python submit.py --dry-run
    python submit.py -- --max-steps 3       # args after -- go to run.sh
"""

from __future__ import annotations

import argparse
import getpass
import subprocess
import sys

import ckpt_tools as C
import settings as S


def queued_jobs() -> dict[str, str]:
    out = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%j %T"], capture_output=True, text=True).stdout
    return {line.split()[0]: line.split()[1] for line in out.splitlines() if line.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=",".join(S.TASKS))
    ap.add_argument("--seeds", default=",".join(map(str, S.SEEDS)))
    ap.add_argument("--partition", default=S.SLURM["partition"])
    ap.add_argument("--time", default=S.SLURM["time"])
    ap.add_argument("--order", choices=("seed", "task"), default="seed", help="seed: a seed for all tasks first; task: all seeds of a task first")
    ap.add_argument("--max-queued", type=int, default=3, help="cap on this tool's jobs that are pending or running")
    ap.add_argument("--sbatch-args", default="", help="extra sbatch arguments, e.g. '--nodelist=<node>'")
    ap.add_argument("--dry-run", action="store_true")
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    args = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)

    tasks, seeds = args.tasks.split(","), [int(s) for s in args.seeds.split(",")]
    jobs = queued_jobs()
    run_names = {S.job_name(t, sd) for t, sd in S.matrix()}
    active = sum(1 for n, st in jobs.items() if n in run_names and st in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"})
    run_sh = S.HERE / "run.sh"
    submitted = 0
    for task, seed in S.matrix(tasks, seeds, args.order):
        name, run = S.job_name(task, seed), S.run_dir(task, seed)
        if C.is_complete(run):
            print(f"complete   {name}")
        elif name in jobs:
            print(f"queued     {name} ({jobs[name]})")
        elif active >= args.max_queued:
            print(f"deferred   {name} (max-queued {args.max_queued} reached)")
        else:
            cmd = ["sbatch", "-J", name, *(["-p", args.partition] if args.partition else []), "-t", args.time, "--gres=gpu:8", "--exclusive",
                   f"--cpus-per-task={S.SLURM['cpus']}", "--mem=0", "-o", str(run / "logs" / "slurm-%j.out"),
                   *args.sbatch_args.split(), "--wrap", " ".join(["bash", str(run_sh), task, str(seed), *extra])]
            if args.dry_run:
                print("would run  " + " ".join(cmd))
            else:
                (run / "logs").mkdir(parents=True, exist_ok=True)
                r = subprocess.run(cmd, capture_output=True, text=True)
                print(f"submitted  {name}: {r.stdout.strip() or r.stderr.strip()}")
                if r.returncode == 0:
                    active += 1
                    submitted += 1
    print(f"{submitted} submitted, {active} active")


if __name__ == "__main__":
    main()
