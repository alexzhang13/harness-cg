"""Env loading (verifiers.load_environment, as prime-rl does) and rollout groups.

Failed or empty rollouts are rescheduled; a group that keeps failing is dropped, as in prime-rl.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import verifiers as vf

from .config import EnvSpec

logger = logging.getLogger(__name__)

STATE_COLUMNS = ["trajectory", "sampling_args"]


@dataclass
class LoadedEnv:
    spec: EnvSpec
    env: Any
    rows: list[dict[str, Any]]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def group_scoring(self) -> bool:
        rubric = getattr(self.env, "rubric", None)
        if rubric is None or not hasattr(rubric, "_get_reward_funcs"):
            return False
        return any(rubric._is_group_func(f) for f in rubric._get_reward_funcs())


def _load(spec: EnvSpec) -> Any:
    return vf.load_environment(spec.id.split("@")[0], **spec.args)


def load_train_env(spec: EnvSpec, seq_len: int, seed: int | None = None) -> LoadedEnv:
    env = _load(spec)
    env.set_kwargs(max_seq_len=seq_len, max_total_completion_tokens=-1)
    ds = env.get_dataset(seed=seed)
    if "example_id" not in ds.column_names:
        ds = ds.map(lambda ex, i: {**ex, "example_id": i}, with_indices=True)
    rows = [dict(r, env_name=spec.name) for r in ds]
    return LoadedEnv(spec, env, rows)


def load_eval_env(spec: EnvSpec) -> LoadedEnv:
    env = _load(spec)
    env.set_kwargs(max_total_completion_tokens=-1)
    rows = [dict(r, env_name=spec.name) for r in env.get_eval_dataset(n=spec.num_examples).to_list()]
    return LoadedEnv(spec, env, rows)


def _failure(output: dict[str, Any]) -> str | None:
    if output.get("error") is not None:
        err = output["error"]
        return err.get("error_chain_repr", str(err)) if isinstance(err, dict) else str(err)
    if not output.get("trajectory"):
        return "empty trajectory"
    return None


async def run_one(le: LoadedEnv, example: dict[str, Any], client: Any, model: str,
                  sampling_args: dict[str, Any]) -> dict[str, Any]:
    out = await le.env.run_rollout(vf.RolloutInput(**example), client=client, model=model,
                                   sampling_args=dict(sampling_args), max_retries=le.spec.max_retries,
                                   state_columns=STATE_COLUMNS)
    out["env_name"] = le.name
    return out


async def run_group(le: LoadedEnv, example: dict[str, Any], n: int, client: Any, model: str,
                    max_attempts: int) -> list[dict[str, Any]] | None:
    """n successful rollouts of one example, or None if dropped."""
    done: list[dict[str, Any]] = []
    failed_rounds = 0
    while len(done) < n:
        need = n - len(done)
        if le.group_scoring:
            outs = await le.env.run_group([vf.RolloutInput(**example) for _ in range(n)], client=client, model=model,
                                          sampling_args=dict(le.spec.sampling_args),
                                          max_retries=le.spec.max_retries, state_columns=STATE_COLUMNS)
            for o in outs:
                o["env_name"] = le.name
        else:
            outs = await asyncio.gather(*[run_one(le, example, client, model, le.spec.sampling_args)
                                          for _ in range(need)])
        reasons = [r for r in map(_failure, outs) if r is not None]
        if le.group_scoring and reasons:
            done = []
        else:
            done.extend(o for o in outs if _failure(o) is None)
        if reasons:
            failed_rounds += 1
            logger.warning("%s example %s: %d failed rollout(s) (round %d): %s", le.name, example.get("example_id"),
                           len(reasons), failed_rounds, reasons[-1][:300])
            if failed_rounds >= max_attempts:
                return None
    return done
