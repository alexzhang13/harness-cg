"""Synthetic env plumbing: rows hold only a seed; context, gold and the oracle's facts are generated per rollout.

Keeping the generated payload out of the dataset columns means the policy can't see hidden facts.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train

from . import program as prog
from .segindex import MIN_SEG, SegmentIndex

SPLIT_OFFSET = {"train": 0, "val": 10_000_000, "test": 20_000_000}


@dataclass
class Generated:
    """What a synthetic generator returns for one seed."""

    context: Any                                  # str | list[str]: the REPL `context`
    root_prompt: str
    items: list[tuple[str, str]]                  # (item id, item text exactly as rendered in `context`)
    gold: Any                                     # JSON-safe gold, read by the scorer
    hidden: dict = field(default_factory=dict)    # JSON-safe facts the oracle answers from
    required_ids: list[str] | None = None         # items a correct solution must have read (coverage metric)
    index_min_len: int = MIN_SEG                  # shortest sentence that counts as evidence
    # shared corpus: prebuilt SegmentIndex + stable cache key (items may be [])
    index: SegmentIndex | None = None
    index_key: str | None = None
    # "package.module:callable" building a custom per-rollout index: fn(items, min_len=..., context_texts=...)
    index_factory: str | None = None


_INDEX: OrderedDict[str, SegmentIndex] = OrderedDict()
_INDEX_MAX = 128
_PINNED: dict[str, SegmentIndex] = {}


def _resolve(spec: str) -> Any:
    import importlib

    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def index_for(state: Any) -> SegmentIndex:
    shared = state.get("synth_index_key")
    if shared and shared in _PINNED:
        return _PINNED[shared]
    key = state["synth_key"]
    idx = _INDEX.get(key)
    if idx is None:
        hidden = state["synth_hidden"]
        ctx = (state.get("info") or {}).get("context")
        texts = [ctx] if isinstance(ctx, str) else [c for c in (ctx or []) if isinstance(c, str)]
        factory = _resolve(hidden.get("_index_factory")) if hidden.get("_index_factory") else SegmentIndex
        idx = factory([(i, t) for i, t in hidden["items"]], min_len=int(hidden.get("_index_min_len") or MIN_SEG),
                      context_texts=texts)
        _INDEX[key] = idx
        while len(_INDEX) > _INDEX_MAX:
            _INDEX.popitem(last=False)
    else:
        _INDEX.move_to_end(key)
    return idx


class SynthEnv(rlm_train.RLMTrainEnv):
    """RLMTrainEnv whose rows are seeds. ``generator(info) -> Generated``."""

    def __init__(self, *, generator: Callable[[dict], Generated], hints: prog.HintSpec | None = None,
                 hint: str = "none", hint_prob: float = 1.0, **kwargs: Any):
        self._generator = generator
        self._hints = hints
        self._hint = hint or "none"
        self._hint_prob = float(hint_prob)
        if self._hint != "none" and hints is not None:
            hints.text(self._hint)
        kwargs.setdefault("code_block_feedback", True)  # Section 4 setting: feedback on turns without a code block
        super().__init__(**kwargs)

    async def setup_state(self, state: vf.State) -> None:
        info = dict(state.get("info") or {})
        if info.get("context") in (None, ""):
            gen = self._generator(info)
            info["context"] = gen.context
            info["root_prompt"] = gen.root_prompt
            state["synth_hinted"] = 0
            if self._hints is not None and self._hint != "none":
                from .oracle import rng_for

                if rng_for("hint", info.get("synth_split"), info.get("synth_seed")).random() < self._hint_prob:
                    info["root_prompt"] = prog.add_hint(gen.root_prompt, self._hints.text(self._hint))
                    state["synth_hinted"] = 1
            state["info"] = info
            hidden = dict(gen.hidden)
            hidden["items"] = [list(x) for x in gen.items]
            hidden["gold"] = gen.gold
            hidden["required_ids"] = list(gen.required_ids or [])
            hidden["_index_min_len"] = gen.index_min_len
            if gen.index_factory:
                hidden["_index_factory"] = gen.index_factory
            state["synth_hidden"] = hidden
            state["synth_key"] = f"{info.get('synth_split')}:{info.get('synth_seed')}:{info.get('synth_params')}"
            state["synth"] = None
            if gen.index is not None:
                key = gen.index_key or state["synth_key"]
                _PINNED[key] = gen.index
                state["synth_index_key"] = key
            index_for(state)
        ctx = (state.get("info") or {}).get("context")
        state["synth_context_chars"] = sum(len(str(x)) for x in ctx) if isinstance(ctx, list) else len(str(ctx or ""))
        await super().setup_state(state)


# The context in state["info"] is replaced before scoring, so scorers must read from Generated.hidden.


def oracle_view(state: Any) -> dict | None:
    """hidden payload + its SegmentIndex, for FakeSubLM.respond."""
    hidden = state.get("synth_hidden") if state is not None else None
    if hidden is None:
        return None
    view = dict(hidden)
    view["_index"] = index_for(state)
    return view


def seed_rows(n: int, split: str, params: dict | None = None, start: int = 0) -> Dataset:
    """``n`` rows of seeds for ``split`` (train / val / test seed ranges never overlap)."""
    p = json.dumps(params or {}, sort_keys=True)
    rows = []
    for i in range(start, start + n):
        seed = SPLIT_OFFSET[split] + i
        rows.append({
            "example_id": i,
            "prompt": [{"role": "user", "content": f"synthetic task {split}:{seed}"}],
            "answer": "",
            "info": {"synth_seed": seed, "synth_split": split, "synth_params": p},
        })
    return Dataset.from_list(rows)


def _st(state: Any) -> dict:
    return (state.get("synth") or {}) if state is not None else {}


def _metric(name: str, fn: Callable[[dict, Any], float]):
    async def m(state: Any = None, **_kw: Any) -> float:
        try:
            return float(fn(_st(state), state))
        except Exception:  # noqa: BLE001
            return 0.0
    m.__name__ = name
    return m


def _coverage(st: dict, state: Any) -> float:
    req = (state.get("synth_hidden") or {}).get("required_ids") or []
    if not req:
        return 0.0
    read = st.get("read_ids") or {}
    return sum(1 for r in req if r in read) / len(req)


STANDARD_METRICS = [
    _metric("synth_calls", lambda st, s: st.get("calls", 0)),
    _metric("synth_batches", lambda st, s: st.get("batches", 0)),
    _metric("synth_max_batch", lambda st, s: st.get("max_batch", 0)),
    _metric("synth_overlong", lambda st, s: st.get("overlong", 0)),
    _metric("synth_refusals", lambda st, s: st.get("refusals", 0)),
    _metric("synth_items_read", lambda st, s: st.get("items_read", 0)),
    _metric("synth_items_skipped", lambda st, s: st.get("items_skipped", 0)),
    _metric("synth_max_prompt_chars", lambda st, s: st.get("max_prompt_chars", 0)),
    _metric("synth_mean_prompt_chars", lambda st, s: st.get("prompt_chars", 0) / max(1, st.get("calls", 0))),
    _metric("synth_required_coverage", _coverage),
]


def make_rubric(score: Callable[[str, Any, dict, Any], float], extra_metrics: Iterable[Callable] = (),
                checks: Iterable[prog.Check] = (), reward_mode: str = "outcome", program_weight: float = 0.2,
                gate_floor: float = 0.0, **rubric_kwargs: Any) -> rlm_train.RLMTrainRubric:
    """Rubric: ``score(answer, gold, hidden, state)`` is the outcome; ``reward_mode`` decides whether ``checks`` count."""
    checks = list(checks)
    prog.combine(0.0, checks, {}, reward_mode)

    def outcome_of(state: Any) -> float:
        hidden = (state or {}).get("synth_hidden") or {}
        final = (state or {}).get("rlm_final_answer")
        if final is None:
            return 0.0
        try:
            return float(max(0.0, min(1.0, score(str(final), hidden.get("gold"), hidden, state))))
        except Exception:  # noqa: BLE001
            return 0.0

    async def correctness(state: Any = None, **_kw: Any) -> float:
        out = outcome_of(state)
        if not checks or reward_mode == "outcome":
            return out
        vals = prog.evaluate(checks, state or {})
        return prog.combine(out, checks, vals, reward_mode, program_weight, gate_floor)

    async def outcome(state: Any = None, **_kw: Any) -> float:
        return outcome_of(state)

    async def program_score(state: Any = None, **_kw: Any) -> float:
        return prog.program_score(checks, prog.evaluate(checks, state or {})) if checks else 0.0

    async def synth_hinted(state: Any = None, **_kw: Any) -> float:
        return float((state or {}).get("synth_hinted") or 0)

    def _check_metric(c: prog.Check):
        async def m(state: Any = None, **_kw: Any) -> float:
            v = prog.evaluate([c], state or {})[c.name]
            return 1.0 if v is None else v
        m.__name__ = f"prog_{c.name}"
        return m

    rubric_kwargs.setdefault("min_iterations", 1)
    rubric = rlm_train.RLMTrainRubric(correctness=correctness, weight=1.0, **rubric_kwargs)
    for m in [outcome, program_score, synth_hinted, *STANDARD_METRICS, *[_check_metric(c) for c in checks],
              *extra_metrics]:
        rubric.add_metric(m)
    return rubric


def make_env(*, generator: Callable[[dict], Generated], oracle: Any, score: Callable[..., float], split: str = "train",
             num_examples: int = 2000, params: dict | None = None, max_iterations: int = 20,
             max_sub_calls: int | None = 300, extra_metrics: Iterable[Callable] = (), user_prologue: str | None = None,
             rubric_kwargs: dict | None = None, hints: prog.HintSpec | None = None, hint: str = "none",
             hint_prob: float = 1.0, checks: Iterable[prog.Check] = (), reward_mode: str = "outcome",
             program_weight: float = 0.2, gate_floor: float = 0.0, verify_checks: bool = False,
             **env_kwargs: Any) -> SynthEnv:
    """Build a synthetic env. ``hint`` / ``hint_prob`` inject hints; ``reward_mode`` puts program checks in the reward."""
    checks = list(checks) + ([prog.delegated(), prog.verified_late(hard=True)] if verify_checks else [])
    rubric = make_rubric(score, extra_metrics, checks=checks, reward_mode=reward_mode, program_weight=program_weight,
                         gate_floor=gate_floor, **(rubric_kwargs or {}))
    return SynthEnv(
        generator=generator,
        hints=hints,
        hint=hint,
        hint_prob=hint_prob,
        dataset=seed_rows(num_examples, split, params),
        max_iterations=max_iterations,
        sub_llm_fn=oracle,
        sub_llm_fn_batched=oracle.batched,
        rubric=rubric,
        max_sub_calls=max_sub_calls,
        user_prologue=user_prologue,
        **env_kwargs,
    )


__all__ = ["Generated", "SynthEnv", "STANDARD_METRICS", "index_for", "make_env", "make_rubric", "oracle_view",
           "seed_rows"]
