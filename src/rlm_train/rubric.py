"""Rubric for RLMTrainEnv: correctness of state["rlm_final_answer"] plus harness metrics.

min_iterations / min_subcall / min_reward only affect the trained reward with gate_reward=True.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import verifiers as vf
from verifiers.types import State


class RLMTrainRubric(vf.Rubric):
    def __init__(
        self,
        correctness: Callable[..., float] | None = None,
        weight: float = 1.0,
        min_iterations: int = 2,
        min_subcall: int = 0,
        min_reward: float = 0.0,
        gate_reward: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._min_iterations = int(min_iterations)
        self._min_subcall = int(min_subcall)
        self._min_reward = float(min_reward)
        self._gate_reward = bool(gate_reward)
        if correctness is not None:
            self.add_reward_func(self._make_main_correctness(correctness), weight=weight)
            self.add_metric(self._make_gated_metric(correctness))
        self.add_metric(self.rlm_iterations)
        self.add_metric(self.rlm_repl_calls)
        self.add_metric(self.rlm_sub_llm_calls)
        self.add_metric(self.rlm_has_final_answer)
        self.add_metric(self.rlm_below_min_iterations)
        self.add_metric(self.rlm_below_min_subcall)
        self.add_metric(self.rlm_below_min_reward)

    def _passes_gates(self, state: dict) -> bool:
        iters = int(state.get("rlm_iterations") or 0)
        if iters < self._min_iterations:
            return False
        sub_calls = int(state.get("rlm_sub_llm_calls") or 0)
        if sub_calls < self._min_subcall:
            return False
        return True

    async def _call_correctness(
        self, correctness: Callable[..., float], kwargs: dict[str, Any]
    ) -> float:
        result = correctness(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return float(result)

    def _make_main_correctness(
        self, correctness: Callable[..., float]
    ) -> Callable[..., Any]:
        """Raw score, or gated score with gate_reward=True."""
        gated_mode = self._gate_reward
        min_rew = self._min_reward

        async def main(**kwargs: Any) -> float:
            state = kwargs.get("state") or {}
            value = await self._call_correctness(correctness, kwargs)
            state["_raw_correctness"] = float(value)
            if not gated_mode:
                return value
            if not self._passes_gates(state):
                return 0.0
            if value < min_rew:
                return 0.0
            return value

        main.__name__ = getattr(correctness, "__name__", "correctness")
        return main

    def _make_gated_metric(
        self, correctness: Callable[..., float]
    ) -> Callable[..., Any]:
        """The gated score, as a metric."""
        min_rew = self._min_reward

        async def gated_reward(**kwargs: Any) -> float:
            state = kwargs.get("state") or {}
            if not self._passes_gates(state):
                return 0.0
            value = await self._call_correctness(correctness, kwargs)
            if value < min_rew:
                return 0.0
            return value

        gated_reward.__name__ = "gated_reward"
        return gated_reward

    async def rlm_iterations(self, state: State) -> int:
        return int(state.get("rlm_iterations") or 0)

    async def rlm_repl_calls(self, state: State) -> int:
        return int(state.get("rlm_repl_calls") or 0)

    async def rlm_sub_llm_calls(self, state: State) -> int:
        return int(state.get("rlm_sub_llm_calls") or 0)

    async def rlm_has_final_answer(self, state: State) -> int:
        return 1 if state.get("rlm_final_answer") else 0

    async def rlm_below_min_iterations(self, state: State) -> int:
        iters = int(state.get("rlm_iterations") or 0)
        return 1 if iters < self._min_iterations else 0

    async def rlm_below_min_subcall(self, state: State) -> int:
        if self._min_subcall <= 0:
            return 0
        sub_calls = int(state.get("rlm_sub_llm_calls") or 0)
        return 1 if sub_calls < self._min_subcall else 0

    async def rlm_below_min_reward(self, state: State) -> int:
        if self._min_reward <= 0.0:
            return 0
        return 1 if float(state.get("_reward") or 0.0) <= 0.0 else 0
