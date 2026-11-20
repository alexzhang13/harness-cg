"""Pi baseline: the `pi` coding agent (verifiers-v1 ``vf.Pi``) run on a host-process sandbox."""

from __future__ import annotations

from .host_pi import (
    PI_SUBAGENT_HEADER,
    PI_SUBAGENT_MODEL_ID,
    ROOT_STEPS_KEY,
    SUBAGENT_STEPS_KEY,
    HostPi,
    count_all_pi_steps,
    keep_only_root_pi_steps,
)
from .process_sandbox import (
    LocalProcessSandboxClient,
    install_local_process_backend,
)
from .v1_env import (
    PI_ANSWER_MARKER,
    PI_CONTEXT_FILENAME,
    compose_instruction,
    last_marked_answer,
    load_backing_env,
    make_pi_env,
    make_source,
    pi_metric,
    pi_reward,
    pi_step_split_metrics,
    row_to_task,
)

__all__ = [
    "LocalProcessSandboxClient",
    "install_local_process_backend",
    "HostPi",
    "PI_ANSWER_MARKER",
    "PI_CONTEXT_FILENAME",
    "PI_SUBAGENT_HEADER",
    "PI_SUBAGENT_MODEL_ID",
    "ROOT_STEPS_KEY",
    "SUBAGENT_STEPS_KEY",
    "compose_instruction",
    "count_all_pi_steps",
    "keep_only_root_pi_steps",
    "last_marked_answer",
    "load_backing_env",
    "make_pi_env",
    "make_source",
    "pi_metric",
    "pi_reward",
    "pi_step_split_metrics",
    "row_to_task",
]
