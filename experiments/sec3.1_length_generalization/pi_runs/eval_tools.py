"""Which Pi checkpoints are evaluated: a checkpoint is done when all ceil(n / UNIT_SIZE) result shards exist."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import ckpt_tools as C
import settings as S

DRIVER = S.HERE.parent / "eval_pi_checkpoints.py"
UNIT_SIZE = 10  # the driver's --unit-size default


def driver_tasks(driver: Path = DRIVER) -> dict[str, int]:
    """{task: n_examples}, read from the driver."""
    for node in ast.parse(driver.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "TASKS" for t in node.targets):
            return {k: int(v["n"]) for k, v in ast.literal_eval(node.value).items()}
    raise RuntimeError(f"TASKS not found in {driver}")


def driver_unit_size(driver: Path = DRIVER) -> int:
    m = re.search(r'"--unit-size",\s*type=int,\s*default=(\d+)', driver.read_text())
    return int(m.group(1)) if m else UNIT_SIZE


def n_shards(task: str, tasks: dict[str, int] | None = None, unit_size: int = UNIT_SIZE) -> int:
    n = (tasks or driver_tasks())[task]
    return (n + unit_size - 1) // unit_size


def adapter_steps(run: Path) -> list[int]:
    """Steps with a complete adapter, ascending."""
    out = []
    for p in (run / "adapters").glob("step_*"):
        suffix = p.name.split("_", 1)[1]
        if suffix.isdigit() and int(suffix) % S.CKPT_INTERVAL == 0 and C.adapter_ok(p):
            out.append(int(suffix))
    return sorted(out)


def shards_done(run: Path, step: int, shards: int) -> int:
    """Number of existing result shards."""
    d = run / "eval" / "results" / f"step_{step}"
    return sum(1 for i in range(shards) if (d / f"shard_{i}.jsonl").is_file())


def eval_state(run: Path, shards: int) -> dict:
    """{'adapters': [...], 'evaluated': [...], 'pending': {step: shards_missing}, 'remaining_units': int}."""
    steps = adapter_steps(run)
    pending = {s: shards - shards_done(run, s, shards) for s in steps}
    pending = {s: m for s, m in pending.items() if m > 0}
    return {"adapters": steps, "evaluated": [s for s in steps if s not in pending], "pending": pending,
            "remaining_units": sum(pending.values())}


def fully_evaluated(run: Path, shards: int) -> bool:
    """All checkpoints 10..150 have an adapter and all shards."""
    st = eval_state(run, shards)
    return len(st["evaluated"]) == len(S.EXPECTED_STEPS)


def step_stats(run: Path, step: int) -> dict:
    """Mean reward, rollouts, errors, cap hits and max root steps of one checkpoint."""
    rows = []
    for f in sorted((run / "eval" / "results" / f"step_{step}").glob("shard_*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    rewards = [float(r.get("reward") or 0.0) for r in rows]
    metrics = [r.get("metrics") or {} for r in rows]
    return {
        "n": len(rows),
        "mean": sum(rewards) / len(rewards) if rewards else None,
        "errors": sum(1 for r in rows if r.get("error")),
        "cap_hits": sum(1 for m in metrics if float(m.get("pi_turn_cap_hit") or 0) > 0),
        "max_root_steps": max((float(m.get("pi_root_steps") or 0) for m in metrics), default=0.0),
    }
