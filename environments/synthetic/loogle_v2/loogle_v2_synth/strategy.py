"""Synthetic LooGLE strategy env: procedural filings, repos, game logs and version trees, with a real sub-LM.

No real LooGLE data is loaded. ``reward_mode="gated"`` (default) gives the outcome only when every ``Budgets`` limit
is met; ``note`` is prepended in ``<strategy_hint>`` tags.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from synth_common.envkit import SynthEnv, seed_rows

from rlm_train import RLMTrainRubric

from .env import make_generator, score
from .finance import FinParams
from .versions import generate_versions


@dataclass(frozen=True)
class Budgets:
    turns: int = 8
    observation_chars: int = 16000
    sub_prompt_chars: int = 32000
    sub_total_chars: int = 160000
    sub_calls: int = 24
    root_total_tokens: int = 64000
    root_peak_tokens: int = 20000
    root_total_chars: int = 240000
    root_peak_chars: int = 80000
    require_source_call: bool = True
    max_repl_errors: int = 0


def measurements(state: dict) -> dict:
    hist = state.get("rlm_history") or []
    observations = []
    for m in hist:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
        if role == "user" and content.startswith("REPL output"):
            observations.append(content)
    final = state.get("rlm_final_repl_outputs") or []
    for m in final:
        c = (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
        if c and c not in observations:
            observations.append(c)
    logs = state.get("rlm_sub_llm_log") or []
    root_inputs = []
    root_outputs = []
    root_prompt_lengths = []
    for step in state.get("trajectory") or []:
        prompt = step.get("prompt", []) if isinstance(step, dict) else getattr(step, "prompt", [])
        root_prompt_lengths.append(sum(len(str((m.get("content") if isinstance(m, dict)
                                               else getattr(m, "content", "")) or "")) for m in prompt))
        response = step.get("response") if isinstance(step, dict) else getattr(step, "response", None)
        usage = response.get("usage", {}) if isinstance(response, dict) else getattr(response, "usage", None)
        if usage:
            root_inputs.append(int(usage.get("prompt_tokens") or 0) if isinstance(usage, dict)
                               else int(getattr(usage, "prompt_tokens", 0) or 0))
            root_outputs.append(int(usage.get("completion_tokens") or 0) if isinstance(usage, dict)
                                else int(getattr(usage, "completion_tokens", 0) or 0))
    return {
        "turns": int(state.get("rlm_iterations") or 0),
        "observation_chars": sum(map(len, observations)),
        "sub_prompt_chars": max((int(x.get("prompt_chars") or 0) for x in logs), default=0),
        "sub_total_chars": int(state.get("rlm_sub_llm_prompt_chars") or 0),
        "sub_calls": int(state.get("rlm_sub_llm_calls") or 0),
        "source_calls": int(state.get("strategy_source_calls") or 0),
        "sub_response_chars": int(state.get("rlm_sub_llm_response_chars") or 0),
        "repl_errors": sum("Traceback (most recent call last)" in c for c in observations),
        "root_total_tokens": sum(root_inputs) + sum(root_outputs),
        "root_peak_tokens": max(root_inputs, default=0),
        "root_total_chars": sum(root_prompt_lengths),
        "root_peak_chars": max(root_prompt_lengths, default=0),
    }


def accepts(state: dict, budgets: Budgets) -> tuple[bool, list[str]]:
    m = measurements(state)
    reasons = [k for k in ("turns", "observation_chars", "sub_prompt_chars", "sub_total_chars", "sub_calls",
                           "root_total_tokens", "root_peak_tokens")
               + ("root_total_chars", "root_peak_chars")
               if m[k] > getattr(budgets, k)]
    if budgets.require_source_call and not m["source_calls"]:
        reasons.append("no_source_call")
    if m["repl_errors"] > budgets.max_repl_errors:
        reasons.append("repl_errors")
    if not str(state.get("rlm_final_answer") or "").strip():
        reasons.append("empty_answer")
    return not reasons, reasons


def source_in_prompt(prompt: str, context: str) -> bool:
    """An actual source span, rather than an id, greeting or generic question."""
    return any(len(s := prompt[i:i+96]) == 96 and s.strip() and s in context
               for i in range(0, min(len(prompt), 65536), 64))


class StrategyEnv(SynthEnv):
    async def setup_state(self, state):
        await super().setup_state(state)
        context = state["info"]["context"]
        proxy = await self._ensure_proxy()
        handle = proxy._handles[state["rlm_rollout_id"]]
        original = handle.record_call

        def record(meta):
            if original:
                original(meta)
            prompt, response = str(meta.get("prompt") or ""), str(meta.get("response") or "")
            if response.strip() and not response.startswith("Error:") and source_in_prompt(prompt, context):
                state["strategy_source_calls"] = int(state.get("strategy_source_calls") or 0) + 1

        handle.record_call = record


def load_strategy_environment(*, split: str = "train", family: str = "mix", num_examples: int = 2000,
                              start_seed: int = 0, note: str = "", budgets: dict | None = None,
                              reward_mode: str = "gated", max_iterations: int = 12,
                              fin: dict | None = None, code: dict | None = None,
                              game: dict | None = None, versions: dict | None = None,
                              generator_version: int = 2, **kwargs: Any):
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    if num_examples <= 0:
        raise ValueError("procedural data needs a positive num_examples")
    if family not in {"mix", "finance", "code", "game", "versions"}:
        raise ValueError("unknown family")
    if reward_mode not in {"gated", "outcome"}:
        raise ValueError("reward_mode must be gated or outcome")
    if generator_version != 2:
        raise ValueError("generator_version must be 2")
    budget = Budgets(**(budgets or {}))

    def outcome(state):
        h = state.get("synth_hidden") or {}
        final = state.get("rlm_final_answer")
        return score(str(final), h.get("gold"), h, state) if final is not None else 0.

    async def correctness(state=None, **_):
        state = state or {}
        good, _ = accepts(state, budget)
        return outcome(state) if reward_mode == "outcome" or good else 0.

    rubric = RLMTrainRubric(correctness=correctness, weight=1., min_iterations=1)
    async def raw_outcome(state=None, **_):
        return outcome(state or {})
    async def efficient(state=None, **_):
        return float(accepts(state or {}, budget)[0])
    rubric.add_metric(raw_outcome)
    rubric.add_metric(efficient)
    for key in measurements({}):
        def metric_for(name):
            async def metric(state=None, **_):
                return float(measurements(state or {})[name])
            metric.__name__ = f"strategy_{name}"
            return metric
        rubric.add_metric(metric_for(key))
    generators = {fam: make_generator(fam, fin=FinParams(**(fin or {})), code_kw=code, game_kw=game)
                  for fam in ("finance", "code", "game")}
    def generate(info):
        fam = family
        if fam == "mix":
            rng = random.Random(f"strategy-family:{info['synth_split']}:{info['synth_seed']}")
            fam = rng.choices(["finance", "code", "game", "versions"], weights=[6, 4, 2, 1])[0]
        return generate_versions(info, **(versions or {})) if fam == "versions" else generators[fam](info)

    return StrategyEnv(generator=generate,
                       dataset=seed_rows(num_examples, split, {"family": family, "budgets": asdict(budget),
                                                              "fin": fin, "code": code, "game": game,
                                                              "versions": versions, "generator_version": generator_version}, start=start_seed),
                       rubric=rubric, max_iterations=max_iterations, max_sub_calls=budget.sub_calls,
                       user_prologue=f"<strategy_hint>\n{note}\n</strategy_hint>" if note else None,
                       sub_extra_body={"chat_template_kwargs": {"enable_thinking": False}}, **kwargs)
