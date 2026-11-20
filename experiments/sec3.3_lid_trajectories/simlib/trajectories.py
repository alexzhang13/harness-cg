"""Trajectories from prime-rl rollout files: a rollout's completion (assistant turns, REPL output, Pi tool calls)
joined into one string, without the prompt or context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


@dataclass
class Traj:
    uid: str
    run_id: str
    split: str
    step: int
    row: int
    example_id: int
    reward: float
    num_turns: int
    rlm_iterations: int
    n_msgs: int
    full_text: str


def _render_tool_calls(m: dict) -> str:
    """tool_calls as text."""
    calls = m.get("tool_calls") or []
    out: list[str] = []
    for c in calls:
        if isinstance(c, str):
            try:
                c = json.loads(c)
            except json.JSONDecodeError:
                out.append(c)
                continue
        if not isinstance(c, dict):
            out.append(str(c))
            continue
        fn = c.get("function") if isinstance(c.get("function"), dict) else c
        name = fn.get("name") or c.get("name") or "tool"
        args = fn.get("arguments", c.get("arguments", ""))
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False, sort_keys=True)
        out.append(f"{name}({args})")
    return "\n".join(out)


def _join_completion(
    completion: list[dict],
    *,
    include_roles: tuple[str, ...] = ("assistant", "user", "tool"),
) -> str:
    parts: list[str] = []
    for m in completion:
        role = m.get("role")
        if role not in include_roles:
            continue
        content = m.get("content") or ""
        if role == "assistant":
            tc = _render_tool_calls(m)
            if tc:
                content = f"{content}\n{tc}" if content else tc
        parts.append(f"<{role}>\n{content}")
    return "\n".join(parts)


def iter_rollout_file(path: Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_file(
    path: Path,
    run_id: str,
    split: str,
    step: int,
    *,
    reward_filter: Callable[[float], bool] | None = None,
    **join_kwargs,
) -> list[Traj]:
    """Trajectories of one rollout file."""
    out: list[Traj] = []
    for row, rec in enumerate(iter_rollout_file(path)):
        reward = float(rec.get("reward", 0.0) or 0.0)
        if reward_filter is not None and not reward_filter(reward):
            continue
        comp = rec.get("completion") or []
        out.append(
            Traj(
                uid=f"{run_id}/{split}@{step}/{row}",
                run_id=run_id,
                split=split,
                step=step,
                row=row,
                example_id=int(rec.get("example_id", -1) or -1),
                reward=reward,
                num_turns=int(rec.get("num_turns", 0) or 0),
                rlm_iterations=int(rec.get("rlm_iterations", 0) or 0),
                n_msgs=len(comp),
                full_text=_join_completion(comp, **join_kwargs),
            )
        )
    return out


def _rollout_root(entry: dict) -> Path:
    root = entry.get("rollout_root")
    if not root:
        raise ValueError(f"run {entry.get('run_id')} has no rollout_root (present={entry.get('present')})")
    return Path(root)


def _eval_step_file(entry: dict, step: int) -> Path:
    return _rollout_root(entry) / f"step_{step}" / "eval_rollouts.jsonl"


def resolve_eval_step(entry: dict) -> int:
    """The selected eval step, or the nearest one with an eval file."""
    avail = entry["eval"]["available_traj_steps"]
    if not avail:
        raise ValueError(f"{entry['run_id']}: no eval trajectory files on disk")
    best = entry["eval"].get("best_step")
    if best is None:
        raise ValueError(f"{entry['run_id']}: best_step not set")
    return best if best in avail else min(avail, key=lambda s: abs(s - best))


def build_eval_pool(entry: dict, **join_kwargs) -> list[Traj]:
    step = resolve_eval_step(entry)
    return extract_file(_eval_step_file(entry, step), entry["run_id"], "eval", step, **join_kwargs)


def build_train_pool(entry: dict, steps: list[int] | None = None, **join_kwargs) -> list[Traj]:
    """Training trajectories with reward > 0."""
    root = _rollout_root(entry)
    if steps is None:
        steps = entry["train"]["available_steps"]
    rf = lambda r: r > 0  # noqa: E731
    out: list[Traj] = []
    for s in steps:
        f = root / f"step_{s}" / "train_rollouts.jsonl"
        if f.exists():
            out.extend(extract_file(f, entry["run_id"], "train", s, reward_filter=rf, **join_kwargs))
    return out
