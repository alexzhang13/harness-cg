"""Shared toolkit for fast synthetic RLM training environments (environments/synthetic/README.md)."""

from .envkit import (
    STANDARD_METRICS,
    Generated,
    SynthEnv,
    index_for,
    make_env,
    make_rubric,
    oracle_view,
    seed_rows,
)
from .oracle import (
    FakeSubLM,
    ReadModel,
    mentions_any,
    overlong_error,
    refuse_aggregate,
    rng_for,
    stable_hash,
    wants_json,
)
from .program import Check, HintSpec, strip_hint
from .segindex import SegmentIndex, segments

__all__ = [
    "Check", "HintSpec", "strip_hint",
    "FakeSubLM", "Generated", "ReadModel", "STANDARD_METRICS", "SegmentIndex", "SynthEnv", "index_for", "make_env",
    "make_rubric", "mentions_any", "oracle_view", "overlong_error", "refuse_aggregate", "rng_for", "seed_rows",
    "segments", "stable_hash", "wants_json",
]
