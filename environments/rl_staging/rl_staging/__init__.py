"""rlstage-* env ids: the 8 synthetic envs with hints off, for RL, plus the OOLONG-real validation env.

Each wrapper pins arguments and calls the package's loader. Passing a hint argument raises unless allow_hint=True.
"""
from __future__ import annotations

import os
from typing import Any

# seed ranges per split, for envs without a `split` argument
SEED = {"train": 100_000, "val": 900_000}

_HINTY = ("hint", "note", "user_prologue", "strategy_hint", "hints")


def _merge(base: dict[str, Any], ov: dict[str, Any], allow_hint: bool) -> dict[str, Any]:
    bad = [k for k, v in ov.items() if k in _HINTY and v not in (None, "", "none", 0, 0.0, False)]
    if bad and not allow_hint:
        raise ValueError(f"rl_staging envs are unhinted by contract; got {bad} (pass allow_hint=True to override)")
    return {**base, **ov}


def load_bcp(split: str = "train", num_examples: int = 2000, allow_hint: bool = False, **ov: Any):
    """BrowseComp-Plus stand-in, fake sub-LM."""
    from browsecomp_plus_synth import load_environment
    return load_environment(**_merge(dict(split=split, num_examples=num_examples, hint="none", hint_prob=0.0,
                                          reward_mode="outcome"), ov, allow_hint))


# families with rows left after decontamination
CTC_FAMILIES = "outlier,contradiction_iid,grouping,textgroups,reorder,qdmatch_hpqa,hotpotqa,nq"


def load_ctc(split: str = "train", num_examples: int = -1, per_family_examples: int | None = None, seed: int = 7,
             allow_hint: bool = False, **ov: Any):
    """CTC-Bench train rows recombined to long rungs, fake sub-LM."""
    from ctcbench_synth import load_environment
    return load_environment(**_merge(dict(split=split, family=CTC_FAMILIES, rungs="native", num_examples=num_examples,
                                          per_family_examples=per_family_examples, seed=seed, hint="none",
                                          hint_prob=0.0, reward_mode="outcome"), ov, allow_hint))


def load_elb(split: str = "train", num_examples: int = 4000, allow_hint: bool = False, **ov: Any):
    """ExpertLongBench mix (docket / recipe / tutor / visit, weighted by SFT row counts), fake sub-LM."""
    from expertlongbench_synth.mix import load_environment
    base = dict(split=split, num_examples=num_examples, p_docket=0.275, form={"p_recipe": 0.345, "p_tutor": 0.54},
                hint="none", reward_mode="outcome")
    return load_environment(**_merge(base, ov, allow_hint))


def load_corpusqa(split: str = "train", num_examples: int = 2000, scale: str = "1m", allow_hint: bool = False,
                  **ov: Any):
    """CorpusQA stand-in, real sub-LM. One stream per scale (1m / 128k)."""
    from corpusqa_synth import load_environment
    return load_environment(**_merge(dict(split=split, num_examples=num_examples, family="periodic", surface="varied",
                                          scale=scale, sub_lm="real", hint="none", hint_prob=0.0,
                                          reward_mode="outcome"), ov, allow_hint))


def load_loogle(split: str = "train", num_examples: int = 2000, family: str = "mix", allow_hint: bool = False,
                **ov: Any):
    """LooGLE-v2 stand-in, real sub-LM, outcome reward (the package default is gated)."""
    from loogle_v2_synth.strategy import load_strategy_environment
    return load_strategy_environment(**_merge(dict(split=split, family=family, num_examples=num_examples, note="",
                                                   reward_mode="outcome", generator_version=2, max_iterations=30),
                                              ov, allow_hint))


def load_oolong(split: str = "train", num_examples: int = 2000, n_episodes: int = 3, surfaces: str = "v2",
                allow_hint: bool = False, **ov: Any):
    """OOLONG-Real stand-in (tabletop_synth), real sub-LM."""
    from tabletop_synth import load_environment
    return load_environment(**_merge(dict(num_examples=num_examples, seed=SEED[split], n_episodes=n_episodes,
                                          surfaces=surfaces, max_iterations=20, max_sub_calls=600), ov, allow_hint))


def load_longreason(split: str = "train", num_examples: int = -1, allow_hint: bool = False, **ov: Any):
    """LongReason stand-in, real sub-LM. val = every 10th item."""
    from longreason_expanded import load_environment
    return load_environment(**_merge(dict(split=split, num_examples=num_examples, length="128k", max_iterations=20,
                                          sub_max_tokens=8192, min_subcall=0), ov, allow_hint))


def load_obliq(split: str = "train", num_examples: int = 2000, allow_hint: bool = False, **ov: Any):
    """OBLIQ stand-in (trick_synth), fake sub-LM with 15% reply faults, nDCG@50."""
    from trick_synth import load_environment
    read = {"preset": "real_v1", "fault_q": 0.15, "fault_desc": 0.15, "fault_flag": 0.15, **ov.pop("read", {})}
    return load_environment(**_merge(dict(split=split, num_examples=num_examples, read=read, max_iterations=20,
                                          max_sub_calls=400, hint="none", hint_prob=0.0), ov, allow_hint))


def load_oolong_real_val(path: str | None = None, num_examples: int = 25,
                         max_episodes: int | None = 3, seed: int = 0, cache_dir: str | None = None, **ov: Any):
    """Validation only: real OOLONG-Real D&D, upstream validation split, one row per window of <= max_episodes episodes.
    The selected rows are cached next to the ~9 GB file."""
    import json
    from pathlib import Path
    from oolong_real_synth.real_task import load_environment, select_rows
    if path is None:
        path = os.path.join(os.environ.get("OOLONG_REAL_DATA_DIR") or os.path.expanduser("~/oolong_real_data"),
                            "dnd", "validation.jsonl")
    cache = Path(cache_dir or Path(path).parent) / f"val_rows_n{num_examples}_ep{max_episodes}_s{seed}.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
    else:
        rows = select_rows(path, n=num_examples, max_episodes=max_episodes, seed=seed, per_window=1)
        cache.write_text(json.dumps(rows))
    return load_environment(rows=rows, **ov)
