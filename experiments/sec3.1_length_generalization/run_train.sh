#!/usr/bin/env bash
# Train one method on one task of the length-generalization experiment (Fig. length-gen).
#
#   [SEED=42] [BACKEND=tinker] bash run_train.sh <method> <task> [extra `rl` args]
#     method: rlm | rlm_orchestrator_prompt | lm_repl | pi | transformer
#     task:   mrcr | graphwalks | longbenchpro | oolong | oolong-pairs | adaleval
#
# SEED (default 42) seeds orchestrator.seed, orchestrator.buffer.seed (example order), inference.seed (vLLM) and torch
# (LoRA init). A seed other than 42 writes to <output_dir>-seed<SEED>. The env `seed = 42` that picks each split's
# examples is not changed.
#
# Pi on MRCRv2, LongBench-Pro, OOLONG-Pairs and Ada-LEval runs through pi_runs/run.sh. BACKEND=tinker trains on Tinker.
set -euo pipefail
METHOD=${1:?method}; TASK=${2:?task}; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CONFIG="$HERE/configs/$METHOD/$TASK.toml"

export ADALEVAL_DATA_DIR="${ADALEVAL_DATA_DIR:-$HOME/ada_leval_data}"
if [[ "$METHOD" == pi ]]; then
  export PATH="${PI_NODE_BIN:-$HOME/node/bin}:$PATH"
  export VF_LOCAL_SANDBOX_NODE_BIN="${VF_LOCAL_SANDBOX_NODE_BIN:-${PI_NODE_BIN:-$HOME/node/bin}}"
  if [[ -f "$HERE/pi_runs/configs/$TASK/seed0.toml" ]]; then
    CONFIG="$HERE/pi_runs/configs/$TASK/seed0.toml"
    [[ "${BACKEND:-prime-rl}" == tinker ]] || exec bash "$HERE/pi_runs/run.sh" "$TASK" 0 "$@"
  fi
  eval "$(python3 "$HERE/pi_runs/settings.py" env "$TASK")"   # Pi hard caps (turns, sub-agent budget, wall clock)
fi
if [[ "${BACKEND:-prime-rl}" == tinker ]]; then
  exec bash "$REPO/training/tinker/launch.sh" "$CONFIG" "$@"
fi
SEED="${SEED:-42}"
export PRIME_RL_TORCH_SEED="$SEED"
export PYTHONPATH="$REPO/training/prime_rl/seed${PYTHONPATH:+:$PYTHONPATH}"
SEED_ARGS=(--orchestrator.seed "$SEED" --orchestrator.buffer.seed "$SEED" --inference.seed "$SEED")
if [[ "$SEED" != 42 ]]; then
  OUT="$(grep -aoE '^output_dir = "[^"]+"' "$CONFIG" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
  NAME="$(awk '/^\[wandb\]/{w=1;next} /^\[/{w=0} w && /^name = /{print; exit}' "$CONFIG" | sed -E 's/.*"([^"]+)".*/\1/')"
  export OUTPUT_DIR="$OUT-seed$SEED"
  SEED_ARGS+=(--wandb.name "$NAME-seed$SEED")
fi
exec bash "$REPO/training/prime_rl/launch.sh" "$CONFIG" "${SEED_ARGS[@]}" "$@"
