"""Synthetic CTC-Bench: real split_v1 rows, alone or combined toward the long eval lengths, with a fake sub-LM.

Reward is the benchmark's own metric. Rows overlapping the published eval set are dropped or scrubbed (decontam.py).

    load_environment(split="train", family="all")
"""

from __future__ import annotations

import functools
import json
import random
import warnings
from typing import Any

from datasets import Dataset
from synth_common import program as P
from synth_common.envkit import Generated, SynthEnv, make_rubric
from synth_common.oracle import FakeSubLM, rng_for
from synth_common.program import add_hint

from .compose import EVAL_ITEMS, build, pool
from .data import FAMILIES, gold_target, render, resolve, score
from .oracles import oracle_for
from .steering import CHECKS, HINTS
from . import xabsence_para as XP

DEFAULT_MIX: dict[str, dict[str, float]] = {
    "textgroups": {"native": 1.0},
    "hotpotqa": {"native": 0.45, "64k": 0.1, "128k": 0.3, "1m": 0.15},
    "nq": {"native": 0.45, "64k": 0.1, "128k": 0.3, "1m": 0.15},
    "outlier_fixedM": {"native": 0.5, "64k": 0.1, "128k": 0.25, "512k": 0.15},
    "outlier": {"s4k": 0.3, "native": 0.3, "64k": 0.1, "128k": 0.2, "1m": 0.1},
    "grouping": {"native": 0.6, "hik": 0.4},
    "reorder": {"w13": 0.25, "w27": 0.25, "native": 0.5},
    "qdmatch_hpqa": {"native": 0.6, "64k": 0.15, "128k": 0.2, "256k": 0.05},
    "contradiction_iid": {"s2k": 0.15, "s4k": 0.15, "native": 0.3, "64k": 0.1, "128k": 0.2, "1m": 0.1},
    XP.FAMILY: {"native": 0.6, "16k": 0.25, "8k": 0.15},
}
EXTRA_FAMILIES = (XP.FAMILY,)
_XP_ROWS = {"train": 2000, "val": 200}


class CTCSynthEnv(SynthEnv):
    """SynthEnv whose hint (if any) is chosen per row's family by the generator."""

    async def setup_state(self, state):
        await super().setup_state(state)
        if ((state.get("synth_hidden") or {}).get("_hinted")):
            state["synth_hinted"] = 1


class Dispatch(FakeSubLM):
    """One oracle per family; the rollout's family picks it."""

    def __init__(self, reorder_oracle: str = "real", grouping_oracle: str = "real", contradiction_oracle: str = "real"):
        super().__init__()
        self._by = {f: oracle_for(f, reorder_oracle=reorder_oracle, grouping_oracle=grouping_oracle,
                                  contradiction_oracle=contradiction_oracle) for f in FAMILIES}
        self._by[XP.FAMILY] = XP.XAbsenceParaOracle()

    def __call__(self, prompt: str, state: Any) -> str:
        fam = ((state or {}).get("synth_hidden") or {}).get("fam") or "nq"
        return self._by[fam](prompt, state)

    def batched(self, prompts: list[str], state: Any) -> list[str]:
        fam = ((state or {}).get("synth_hidden") or {}).get("fam") or "nq"
        return self._by[fam].batched(prompts, state)


def _slim(ex: dict) -> dict:
    out = {k: v for k, v in ex.items() if k != "documents"}
    out["n_docs"] = len(ex.get("documents") or [])
    return out


def generate(info: dict, hint: str = "none", hint_prob: float = 1.0) -> Generated:
    p = json.loads(info.get("synth_params") or "{}")
    if p["family"] == XP.FAMILY:
        return _generate_xp(info, p)
    fam = resolve(p["family"])
    inst = build(fam, info["synth_split"], int(p["row"]), p.get("rung", "native"), int(info["synth_seed"]))
    hidden = dict(inst.facts)
    hidden["fam"] = fam
    hidden["rung"] = inst.rung
    hidden["ex"] = _slim(inst.ex)
    root_prompt = inst.root_prompt
    hidden["_hinted"] = 0
    if hint not in (None, "", "none") and rng_for("hint", info.get("synth_split"), info.get("synth_seed")).random() < hint_prob:
        root_prompt = add_hint(root_prompt, HINTS[fam].text(hint))
        hidden["_hinted"] = 1
    inst.root_prompt = root_prompt
    return Generated(context=inst.context, root_prompt=inst.root_prompt, items=inst.items,
                     gold=gold_target(inst.ex, fam), hidden=hidden, required_ids=inst.required,
                     index_min_len=12 if fam in ("contradiction_iid", "qdmatch_hpqa") else 24)


def _generate_xp(info: dict, p: dict) -> Generated:
    ex = XP.compose(info["synth_split"], int(p["row"]), p.get("rung", "native"), int(info["synth_seed"]))
    context, root_prompt = render(ex, XP.FAMILY)
    hidden = {"fam": XP.FAMILY, "rung": p.get("rung", "native"), "ex": _slim(ex), "xab": XP.facts(ex), "_hinted": 0}
    items = [(str(i + 1), d["text"]) for i, d in enumerate(ex["documents"])]
    return Generated(context=context, root_prompt=root_prompt, items=items, gold=gold_target(ex, XP.FAMILY),
                     hidden=hidden, required_ids=[str(g + 1) for g in ex["gold_doc_indices"]], index_min_len=12)


def _score(final: str, gold: Any, hidden: dict, state: Any) -> float:
    m = score(final, hidden["ex"], hidden["fam"])
    if state is not None:
        state["_ctc_metrics"] = m
    return m["primary"]


def _metric(name: str, fn):
    async def m(state: Any = None, **_kw: Any) -> float:
        try:
            return float(fn(state))
        except Exception:  # noqa: BLE001
            return 0.0
    m.__name__ = name
    return m


def _ctc(state: Any, key: str) -> float:
    cached = (state or {}).get("_ctc_metrics")
    if cached is None:
        h = (state or {}).get("synth_hidden") or {}
        cached = score((state or {}).get("rlm_final_answer"), h["ex"], h["fam"])
    return cached.get(key, 0.0)


EXTRA_METRICS = [
    _metric("ctc_primary_raw", lambda s: _ctc(s, "primary")),
    _metric("ctc_parse_ok", lambda s: _ctc(s, "parse_ok")),
    _metric("ctc_n_items", lambda s: ((s or {}).get("synth_hidden") or {}).get("ex", {}).get("n_docs", 0)),
    _metric("ctc_long", lambda s: float(((s or {}).get("synth_hidden") or {}).get("rung", "native")
                                       not in ("native", "hik"))),
]


def _mix_for(fam: str, rungs: Any) -> dict[str, float]:
    if rungs is None or rungs == "mix":
        return DEFAULT_MIX[fam]
    if rungs == "native":
        return {"native": 1.0}
    if rungs == "long":
        m = {k: v for k, v in DEFAULT_MIX[fam].items() if k not in ("native",)}
        return m or {"native": 1.0}
    if isinstance(rungs, str):
        rs = [r.strip() for r in rungs.split(",") if r.strip()]
        rs = [r for r in rs if r == "native" or r in EVAL_ITEMS.get(fam, {}) or
              (fam == XP.FAMILY and r in XP.RUNG_PAIRS)]
        return {r: 1.0 for r in rs} or {"native": 1.0}
    if isinstance(rungs, dict):
        if fam in rungs and isinstance(rungs[fam], (dict, str)):
            return _mix_for(fam, rungs[fam])
        m = {r: float(w) for r, w in rungs.items() if r == "native" or r in EVAL_ITEMS.get(fam, {})}
        return m or {"native": 1.0}
    raise ValueError(f"bad rungs={rungs!r}")


def build_rows(split: str, families: list[str], num_examples: int, rungs: Any, seed: int,
               per_family: int | None = None) -> list[dict]:
    rows = []
    rng = random.Random(f"ctcsynth-rows:{split}:{seed}")
    per = per_family or (max(1, num_examples // len(families)) if num_examples > 0 else None)
    k = 0
    for fam in families:
        n_rows = _XP_ROWS[split] if fam == XP.FAMILY else len(pool(fam, split))
        if n_rows == 0:
            warnings.warn(f"ctcbench-synth: family {fam!r} has no eval-clean {split} rows; skipped", stacklevel=2)
            continue
        n = per if per is not None else n_rows
        mix = _mix_for(fam, rungs)
        names, weights = list(mix), [mix[r] for r in mix]
        order = list(range(n_rows))
        rng.shuffle(order)
        for i in range(n):
            rung = rng.choices(names, weights)[0]
            row = order[i % n_rows]
            k += 1
            rows.append({
                "example_id": 0,
                "prompt": [{"role": "user", "content": f"ctc synthetic {fam} {split}:{row}:{rung}"}],
                "answer": "",
                "info": {"synth_seed": seed * 1_000_003 + k, "synth_split": split,
                         "synth_params": json.dumps({"family": fam, "row": row, "rung": rung}, sort_keys=True)},
            })
    rng.shuffle(rows)
    if num_examples > 0 and per_family is None:
        rows = rows[:num_examples]
    for i, r in enumerate(rows):
        r["example_id"] = i
    return rows


def load_environment(
    split: str = "train",
    family: str | list[str] = "all",
    num_examples: int = -1,
    rungs: Any = None,
    per_family_examples: int | None = None,
    seed: int = 0,
    max_iterations: int = 30,
    max_sub_calls: int | None = 1500,
    hint: str = "none",
    hint_prob: float = 1.0,
    reward_mode: str = "outcome",
    program_weight: float = 0.2,
    gate_floor: float = 0.0,
    checks: list | None = None,
    verify_checks: bool = False,
    reorder_oracle: str = "real",
    grouping_oracle: str = "real",
    contradiction_oracle: str = "real",
    real_sub_calls: bool = False,
    **kwargs: Any,
) -> SynthEnv:
    """CTC-Bench synthetic env.

    rungs: "mix" (default), "native", "long", a comma list, or a {rung: weight} dict (optionally per family).
    *_oracle: "real" (error rates of the real 27B sub-LM) or "strong" (more accurate).
    real_sub_calls: False (default, RL training) answers every sub-call with the family's fake oracle. True installs no
        oracle, so sub-calls go to the real sub-LM (thinking off unless `sub_extra_body` says otherwise); rows, prompts
        and scoring are unchanged and the *_oracle arguments are ignored. `max_sub_calls` is then the only fan-out limit.
    """
    fams = list(FAMILIES) if family in ("all", None) else [f.strip() if f.strip() in EXTRA_FAMILIES else resolve(f.strip())
                                                          for f in (family.split(",") if isinstance(family, str) else family)]
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val' (the eval rungs are scored with environments/real/CTCBench)")
    rows = build_rows(split, fams, num_examples, rungs, seed, per_family_examples)
    checks = list(CHECKS if checks is None else checks)
    if verify_checks:
        checks += [P.delegated(), P.verified_late(hard=True)]
    rubric = make_rubric(_score, extra_metrics=EXTRA_METRICS, checks=checks,
                         reward_mode=reward_mode, program_weight=program_weight, gate_floor=gate_floor)
    if real_sub_calls:
        oracle = None
        kwargs.setdefault("sub_extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
    else:
        oracle = Dispatch(reorder_oracle=reorder_oracle, grouping_oracle=grouping_oracle,
                          contradiction_oracle=contradiction_oracle)
    if hint not in ("none", "strategy", "program", "verify"):
        raise ValueError(f"hint must be none | strategy | program | verify, got {hint!r}")
    return CTCSynthEnv(
        generator=functools.partial(generate, hint=hint, hint_prob=hint_prob),
        dataset=Dataset.from_list(rows),
        max_iterations=max_iterations,
        sub_llm_fn=oracle,
        sub_llm_fn_batched=None if oracle is None else oracle.batched,
        rubric=rubric,
        max_sub_calls=max_sub_calls,
        **kwargs,
    )


__all__ = ["DEFAULT_MIX", "build_rows", "generate", "load_environment"]
