"""Status of the Pi runs: state, adapters, last step and SLURM job per run.

    python status.py                     # table
    python status.py --tail longbenchpro 0    # last lines of that run's training log
"""

from __future__ import annotations

import argparse
import getpass
import subprocess
import time
from pathlib import Path

import ckpt_tools as C
import settings as S


def slurm_states() -> dict[str, str]:
    out = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%j %T %M %N"], capture_output=True, text=True).stdout
    return {l.split()[0]: " ".join(l.split()[1:]) for l in out.splitlines() if l.strip()}


def last_log_age(run: Path) -> str:
    log = run / "logs" / "train.log"
    if not log.exists():
        return "-"
    age = time.time() - log.stat().st_mtime
    return f"{age / 60:.0f} min ago" if age < 7200 else f"{age / 3600:.1f} h ago"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", nargs=2, metavar=("TASK", "SEED"))
    args = ap.parse_args()
    if args.tail:
        log = S.run_dir(args.tail[0], int(args.tail[1])) / "logs" / "train.log"
        print("\n".join(log.read_text(errors="ignore").splitlines()[-40:]))
        return
    jobs = slurm_states()
    print(f"{'task':13s} {'seed':>6s} {'state':9s} {'adapters':9s} {'last step':>9s} {'trainer ckpt':>12s}  {'log':12s} slurm")
    counts: dict[str, int] = {}
    for task, seed in S.matrix():
        run = S.run_dir(task, seed)
        st = C.status(run)
        counts[st["state"]] = counts.get(st["state"], 0) + 1
        print(f"{task:13s} {seed:>6d} {st['state']:9s} {st['adapters']:9s} {str(st['last_rollout_step'] or '-'):>9s} "
              f"{str(st['trainer_ckpt_step'] or '-'):>12s}  {last_log_age(run):12s} {jobs.get(S.job_name(task, seed), '')}")
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + f"   (runs root: {S.RUNS_ROOT})")


if __name__ == "__main__":
    main()
