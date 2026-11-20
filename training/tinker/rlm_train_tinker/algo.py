"""Advantages, rollout filters and the example buffer, as in prime-rl@eaf95997f.

Advantage = r - group mean (optional turns length penalty). Filters: gibberish, repetition, zero advantage.
The buffer samples an env by ratio, then an example with replacement, with prime-rl's difficulty pools.
"""

from __future__ import annotations

import math
import random
from typing import Any


def _shaped(rewards: list[float], costs: list[float]) -> list[float]:
    """Correct rollouts get up to 2x reward for below-average cost."""
    mean = sum(rewards) / len(rewards)
    unshaped = [r - mean for r in rewards]
    max_r = max(rewards)
    if max_r <= 0:
        return unshaped
    correct = [r >= max_r for r in rewards]
    n_correct = max(1, sum(correct))
    mean_cost = sum(c for c, ok in zip(costs, correct, strict=True) if ok) / n_correct
    if mean_cost <= 0:
        bonus = [0.0] * len(costs)
    else:
        bonus = [min(1.0, max(0.0, 1 - c / mean_cost)) for c in costs]
    shaped = [r * (1 + b * ok) for r, b, ok in zip(rewards, bonus, correct, strict=True)]
    base = sum(shaped) / len(shaped)
    return [s - base for s in shaped]


def compute_advantages(group: list[dict[str, Any]], length_penalty: dict[str, Any] | None = None) -> None:
    rewards = [float(r["reward"]) for r in group]
    kind = (length_penalty or {}).get("type")
    if kind == "turns":
        advs = _shaped(rewards, [float(len(r.get("trajectory") or [])) for r in group])
    elif kind == "tokens":
        wc = float(length_penalty.get("completion_weight", 1.0))
        wt = float(length_penalty.get("tool_response_weight", 0.0))
        costs = []
        for r in group:
            completion = sum(len((s.get("tokens") or {}).get("completion_ids") or []) for s in r.get("trajectory") or [])
            tool = next((float(v) for k, v in (r.get("metrics") or {}).items()
                         if k.endswith("total_tool_response_tokens") and isinstance(v, (int, float))), 0.0)
            costs.append(wc * completion + wt * tool)
        advs = _shaped(rewards, costs)
    elif kind is None:
        mean = sum(rewards) / len(rewards)
        advs = [r - mean for r in rewards]
    else:
        raise ValueError(f"unknown length_penalty type {kind!r}")
    for r, a in zip(group, advs, strict=True):
        r["advantage"] = a


def _completion_tokens(rollout: dict[str, Any]):
    for step in rollout.get("trajectory") or []:
        t = step.get("tokens")
        if t is not None:
            yield from zip(t["completion_ids"], t["completion_logprobs"], strict=False)


def apply_filters(rollouts: list[dict[str, Any]], filters: list[dict[str, Any]], vocab_size: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rollouts:
        r["is_filtered"] = False
        r["filter"] = None
        for f in filters:
            kind = f["type"]
            if kind == "gibberish":
                id_thr = int(f.get("token_id_threshold", 100000))
                lp_thr = -math.log(vocab_size) - float(f.get("logprob_offset", 2.0))
                hit = any(tid > id_thr and lp < lp_thr for tid, lp in _completion_tokens(r))
            elif kind == "repetition":
                window, thr, run, hit = int(f.get("window", 3000)), math.log(float(f.get("prob_threshold", 0.99))), 0, False
                for _, lp in _completion_tokens(r):
                    run = run + 1 if lp > thr else 0
                    if run >= window:
                        hit = True
                        break
            elif kind == "zero_advantage":
                hit = r.get("advantage") is not None and r["advantage"] == 0.0
            else:
                raise ValueError(f"unknown filter {kind!r}")
            if hit:
                counts[kind] = counts.get(kind, 0) + 1
                r["filter"] = kind
                r["is_filtered"] = bool(f.get("enforce", False))
                break
    return counts


class Buffer:
    """Example sampling by env ratio, with difficulty pools."""

    def __init__(self, env_rows: dict[str, list[dict[str, Any]]], ratios: dict[str, float | None],
                 cfg: dict[str, Any], rng: random.Random):
        self.rng = rng
        self.easy_threshold = cfg.get("easy_threshold")
        self.hard_threshold = cfg.get("hard_threshold")
        self.online_difficulty_filtering = bool(cfg.get("online_difficulty_filtering", False))
        self.pools: dict[str, dict[Any, dict[str, Any]]] = {}
        for name, rows in env_rows.items():
            if not rows:
                raise ValueError(f"train env {name} has no examples")
            self.pools[name] = {row["example_id"]: row for row in rows}
        self.evicted: dict[str, int] = {"easy": 0, "hard": 0}
        if any(v is not None for v in ratios.values()):
            weights = {k: float(v) for k, v in ratios.items()}
        else:
            weights = {k: float(len(v)) for k, v in self.pools.items()}
        total = sum(weights.values())
        self.probs = {k: v / total for k, v in weights.items()}

    def sample_example(self) -> tuple[str, dict[str, Any]]:
        names = [n for n, pool in self.pools.items() if pool]
        if not names:
            raise RuntimeError("no examples left in any train env (all evicted)")
        name = self.rng.choices(names, weights=[self.probs[n] for n in names], k=1)[0]
        pool = self.pools[name]
        return name, pool[self.rng.choice(tuple(pool))]

    def accept(self, env_name: str, example_id: Any, group: list[dict[str, Any]]) -> bool:
        """Update the pools; False if the group is dropped."""
        avg = sum(float(r["reward"]) for r in group) / len(group)
        pool = None
        if self.easy_threshold is not None and avg >= self.easy_threshold:
            pool = "easy"
        elif self.hard_threshold is not None and avg <= self.hard_threshold:
            pool = "hard"
        if pool is not None and example_id in self.pools[env_name]:
            del self.pools[env_name][example_id]
            self.evicted[pool] += 1
        if self.online_difficulty_filtering and avg in (0.0, 1.0):
            return False
        return True
