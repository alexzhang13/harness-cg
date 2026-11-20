"""Instantiate the verifiers env of one stream (thin wrapper over strategies)."""

from __future__ import annotations

from typing import Any

from .strategies import Strategy

TaskDescriptor = Strategy


def load_env(st: Strategy, *, num_examples: int = -1, with_hint: bool = True,
             sub_no_think: bool = False, sub_think: bool = False, **extra: Any):
    """Instantiate the verifiers env for a strategy, hint injected as user_prologue."""
    import verifiers as vf

    args = st.rollout_env_args() if with_hint else {k: v for k, v in st.env_args.items() if k not in ("gate_subcall", "min_subcall")}
    args.pop("num_examples", None)
    if sub_no_think:
        args["sub_extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    if sub_think:
        args["sub_extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    args.update(extra)
    return vf.load_environment(st.env_id, num_examples=num_examples, **args)
