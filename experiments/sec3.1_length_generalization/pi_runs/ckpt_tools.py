"""Checkpoint handling for one run directory: copy adapters to <run>/adapters, pick a resume step, finalize.

Commands: complete | sync | sync-loop | prepare-resume | finalize | status <run dir>
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import settings as S

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
MIN_ADAPTER_BYTES = 1_000_000


def adapter_ok(path: Path) -> bool:
    w = path / ADAPTER_FILES[0]
    return w.is_file() and w.stat().st_size > MIN_ADAPTER_BYTES and (path / ADAPTER_FILES[1]).is_file()


def missing_adapters(run: Path) -> list[int]:
    return [n for n in S.EXPECTED_STEPS if not adapter_ok(run / "adapters" / f"step_{n}")]


def is_complete(run: Path) -> bool:
    return (run / "COMPLETE").is_file() and not missing_adapters(run)


def sync(run: Path) -> list[int]:
    """Copy finished (STABLE) broadcasts into <run>/adapters."""
    copied = []
    for n in S.EXPECTED_STEPS:
        dst = run / "adapters" / f"step_{n}"
        src = run / "run_default" / "broadcasts" / f"step_{n}"
        if adapter_ok(dst) or not (src / "STABLE").is_file() or not adapter_ok(src):
            continue
        tmp = run / "adapters" / f".tmp_step_{n}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for name in ADAPTER_FILES:
            shutil.copy2(src / name, tmp / name)
        shutil.rmtree(dst, ignore_errors=True)
        tmp.rename(dst)
        copied.append(n)
    return copied


def sync_loop(run: Path, interval: float = 30.0) -> None:
    stop = run / ".sync_stop"
    while not stop.exists():
        sync(run)
        time.sleep(interval)


def _steps(path: Path) -> list[int]:
    return sorted(int(p.name.split("_")[1]) for p in path.glob("step_*") if p.name.split("_")[1].isdigit())


def prepare_resume(run: Path) -> int | None:
    """Latest step with both a complete trainer and orchestrator checkpoint, or None.

    prime-rl resumes from the newest step_* without checking it, so newer or incomplete ones are moved to ckpt_partial/.
    """
    trainer_dir, orch_dir = run / "checkpoints", run / "run_default" / "checkpoints"
    valid = [n for n in _steps(trainer_dir)
             if (trainer_dir / f"step_{n}" / "trainer" / ".metadata").is_file()
             and (orch_dir / f"step_{n}" / "orchestrator").is_dir()]
    keep = max(valid) if valid else None
    aside = run / "ckpt_partial" / time.strftime("%Y%m%d-%H%M%S")
    for base in (trainer_dir, orch_dir):
        for n in _steps(base):
            if n != keep:
                aside.mkdir(parents=True, exist_ok=True)
                shutil.move(str(base / f"step_{n}"), str(aside / f"{base.parent.name}-{base.name}-step_{n}"))
    return keep


def finalize(run: Path) -> bool:
    sync(run)
    (run / ".sync_stop").touch()
    missing = missing_adapters(run)
    if missing:
        print(f"[finalize] NOT complete, missing adapters for steps {missing}")
        return False
    (run / "COMPLETE").write_text(json.dumps({"steps": list(S.EXPECTED_STEPS), "finished": time.strftime("%F %T")}) + "\n")
    shutil.rmtree(run / "checkpoints", ignore_errors=True)  # ~58 GB
    shutil.rmtree(run / "run_default" / "checkpoints", ignore_errors=True)
    shutil.rmtree(run / "ckpt_partial", ignore_errors=True)
    print(f"[finalize] complete: {len(S.EXPECTED_STEPS)} adapters in {run / 'adapters'}")
    return True


def status(run: Path) -> dict:
    rollouts = run / "run_default" / "rollouts"
    done = len(S.EXPECTED_STEPS) - len(missing_adapters(run))
    last = max(_steps(rollouts), default=None) if rollouts.is_dir() else None
    state = "complete" if is_complete(run) else ("partial" if run.exists() and (done or last is not None) else "none")
    return {"state": state, "adapters": f"{done}/{len(S.EXPECTED_STEPS)}", "last_rollout_step": last,
            "trainer_ckpt_step": max(_steps(run / "checkpoints"), default=None)}


if __name__ == "__main__":
    cmd, run = sys.argv[1], Path(sys.argv[2])
    if cmd == "complete":
        sys.exit(0 if is_complete(run) else 1)
    elif cmd == "sync":
        print("copied", sync(run))
    elif cmd == "sync-loop":
        sync_loop(run)
    elif cmd == "prepare-resume":
        step = prepare_resume(run)
        print("" if step is None else step)
    elif cmd == "finalize":
        sys.exit(0 if finalize(run) else 1)
    elif cmd == "status":
        print(json.dumps(status(run)))
    else:
        raise SystemExit(f"unknown command: {cmd}")
