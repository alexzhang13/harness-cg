#!/usr/bin/env bash
# "+ RL (synthetic tasks)" rows of Table main-composition: RL on the eight synthetic environments, no hints.
#
#   bash run_train.sh qwen3-30b-a3b                 LoRA; the paper evaluates the step-207 adapter
#   BACKEND=tinker bash run_train.sh qwen3-30b-a3b  the same on Tinker
#   bash run_train.sh qwen38-27b                    full fine-tuning, 150 steps
#
# Validation needs OOLONG_REAL_DATA_DIR (environments/synthetic/oolong_real/README.md).
set -euo pipefail
export RLM_TRAIN_CODE_BLOCK_FEEDBACK=${RLM_TRAIN_CODE_BLOCK_FEEDBACK:-1}  # Section 4 setting: feedback on turns without a code block
MODEL=${1:?model}; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
LAUNCH="$REPO/training/prime_rl/launch.sh"
case "$MODEL" in
  qwen3-30b-a3b)
    [[ "${BACKEND:-prime-rl}" == tinker ]] && LAUNCH="$REPO/training/tinker/launch.sh"
    exec bash "$LAUNCH" "$HERE/configs/qwen3-30b-a3b.toml" "$@" ;;
  qwen38-27b)
    [[ "${BACKEND:-prime-rl}" == tinker ]] && { echo "qwen38-27b is full fine-tuning: prime-rl only" >&2; exit 1; }
    export RLM_MULTI_USER_TAIL=1
    exec bash "$LAUNCH" "$HERE/configs/qwen38-27b.toml" "$@" ;;
  *) echo "unknown model: $MODEL" >&2; exit 1 ;;
esac
