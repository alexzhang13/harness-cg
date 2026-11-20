#!/usr/bin/env bash
# "+ RL (short tasks)" rows of Table main-composition: RL on the six short length-generalization streams.
#
#   [BACKEND=tinker] bash run_train.sh qwen3-30b-a3b     LoRA, 1000 steps (same run as appendix/short_task_mixture rlm)
#   bash run_train.sh qwen38-27b        full fine-tuning, lr 5e-6, batch 96, 150 steps
#
# Evaluate with eval/. Paper checkpoints: 30B step 1000, 27B step 150 (fix its export with fix_export_names.py).
set -euo pipefail
MODEL=${1:?model}; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
case "$MODEL" in
  qwen3-30b-a3b) exec bash "$HERE/../appendix/short_task_mixture/run_train.sh" rlm "$@" ;;
  qwen38-27b)
    [[ "${BACKEND:-prime-rl}" == tinker ]] && { echo "qwen38-27b is full fine-tuning: prime-rl only" >&2; exit 1; }
    export RLM_MULTI_USER_TAIL=1
    export RLM_TRAIN_CODE_BLOCK_FEEDBACK=${RLM_TRAIN_CODE_BLOCK_FEEDBACK:-1}  # Section 4 setting: feedback on turns without a code block
    exec bash "$REPO/training/prime_rl/launch.sh" "$HERE/configs/qwen38-27b.toml" "$@" ;;
  *) echo "unknown model: $MODEL" >&2; exit 1 ;;
esac
