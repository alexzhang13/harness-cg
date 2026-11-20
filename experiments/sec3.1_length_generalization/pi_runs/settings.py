"""Settings of the Pi length-generalization runs (tasks, limits, timeouts, GPU layout, paths), used by every script here."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

TASKS = ("mrcr", "longbenchpro", "oolong-pairs", "adaleval")
SEEDS = (0,)
RUNS_ROOT = Path(os.environ.get("CG_RUNS_ROOT", os.path.join(os.environ.get("PRIME_RL_DIR", str(Path.home() / "prime-rl")), "outputs")))

STEPS = 150
CKPT_INTERVAL = 10
EXPECTED_STEPS = tuple(range(CKPT_INTERVAL, STEPS + 1, CKPT_INTERVAL))

# 6 vLLM GPUs (tensor parallel 2 x data parallel 3) + 2 trainer GPUs on one 8-GPU node.
GPUS = {"total": 8, "infer": 6, "train": 2, "tp": 2, "dp": 3}

# Also set in configs/<task>/seed0.toml.
TRAIN = {
    "max_model_len": 65536,  # one window for every task; compaction off
}

# Pi limits, exported to every run (read by the extensions and quoted in the prompt).
PI_ENV = {
    "PI_PROMPT_VARIANT": "short",
    "PI_MAX_TURNS": 30,  # root model calls
    "PI_SUBAGENT_MAX_CALLS": 100,  # per rollout
    "PI_SUBAGENT_MAX_TURNS": 10,
    "PI_WALL_TIMEOUT_S": 600,  # ends the rollout with reward 0
    "PI_SUBAGENT_WALL_TIMEOUT_S": 120,
    "PI_COMMAND_TIMEOUT": 660,  # backstop, above the wall clock
    "PI_ROLLOUT_TIMEOUT": 720,
    "PI_CONTEXT_WINDOW": TRAIN["max_model_len"],
}

# Per-task additions to PI_ENV. PI_PROMPT_GUIDE adds instructions for reading large files (the 6 KB `read` tool is too
# small for a 1 MB context) and the helper names; PI_TURN_WARN_BEFORE adds turn-budget notices over the last 6 turns;
# PI_PROMPT_TASK adds a task paragraph (format and strategy).
TASK_PI_ENV = {
    "longbenchpro": {"PI_PROMPT_GUIDE": 1, "PI_TURN_WARN_BEFORE": 6},
    # helpers (not the root) may read 96 KB per tool call
    "adaleval": {"PI_PROMPT_GUIDE": 1, "PI_TURN_WARN_BEFORE": 6, "PI_PROMPT_TASK": "adaleval",
                 "PI_SUBAGENT_TOOL_OUTPUT_CAP_BYTES": 96000},
    "oolong-pairs": {"PI_PROMPT_GUIDE": 1, "PI_TURN_WARN_BEFORE": 6, "PI_PROMPT_TASK": "oolong-pairs",
                     "PI_SUBAGENT_TOOL_OUTPUT_CAP_BYTES": 96000},
}

# Per-task training settings (also in the configs). The oolong-pairs answer is a long pair list, hence the larger output
# budget. prime-rl truncates samples at seq_len and a Pi sample is the whole rollout, hence 32,768.
TASK_TRAIN = {
    "oolong-pairs": {"max_completion_tokens": 12288, "seq_len": 32768, "max_retries": 1},
    "adaleval": {"seq_len": 32768},
}

# Per-task overrides of the evaluation limits in slurm/eval_server.sbatch (1800 s root, 300 s helper, 1920 s rollout).
TASK_EVAL_ENV = {
    "oolong-pairs": {"PI_WALL_TIMEOUT_S": 3600, "PI_SUBAGENT_WALL_TIMEOUT_S": 600,
                     "PI_COMMAND_TIMEOUT": 3660, "PI_ROLLOUT_TIMEOUT": 3720},
}


def eval_env_exports(task: str = "") -> str:
    return "\n".join(f"export {k}={v}" for k, v in TASK_EVAL_ENV.get(task, {}).items())


# Evaluation output tokens per model call (default 4,096); the largest oolong-pairs gold list at 32k has 10,440 pairs.
EVAL_MAX_TOKENS = {"oolong-pairs": 131072}


def pi_env(task: str = "") -> dict:
    return {**PI_ENV, **TASK_PI_ENV.get(task, {})}


# empty = cluster default
SLURM = {"partition": os.environ.get("CG_SLURM_PARTITION", ""), "time": "14:00:00", "cpus": 64}


# Suffix for SLURM job names, to keep runs in two CG_RUNS_ROOTs apart (the submit scripts match jobs by name).
JOB_TAG = os.environ.get("CG_JOB_TAG", "")


def run_dir(task: str, seed: int) -> Path:
    return RUNS_ROOT / "length-gen" / "pi" / task / f"seed{seed}"


def job_name(task: str, seed: int) -> str:
    return f"cgpi-{task}-s{seed}{JOB_TAG}"


def matrix(tasks=None, seeds=None, order="seed"):
    """(task, seed) pairs, grouped by seed or by task."""
    tasks, seeds = tasks or TASKS, seeds or SEEDS
    if order == "task":
        return [(t, s) for t in tasks for s in seeds]
    return [(t, s) for s in seeds for t in tasks]


def snapshot(task: str, seed: int) -> dict:
    return {
        "task": task,
        "seed": seed,
        "steps": STEPS,
        "ckpt_interval": CKPT_INTERVAL,
        "gpus": GPUS,
        "train": TRAIN,
        "pi_env": pi_env(task),
    }


def shell_exports(task: str = "") -> str:
    """`export K=${K:-V}` lines (an already exported value wins)."""
    return "\n".join(f"export {k}=${{{k}:-{v}}}" for k, v in pi_env(task).items())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "env":
        print(shell_exports(sys.argv[2] if len(sys.argv) > 2 else ""))
    elif cmd == "eval-env":
        print(eval_env_exports(sys.argv[2] if len(sys.argv) > 2 else ""))
    elif cmd == "run-dir":
        print(run_dir(sys.argv[2], int(sys.argv[3])))
    elif cmd == "snapshot":
        print(json.dumps(snapshot(sys.argv[2], int(sys.argv[3])), indent=1))
    else:
        raise SystemExit(f"unknown command: {cmd}")
