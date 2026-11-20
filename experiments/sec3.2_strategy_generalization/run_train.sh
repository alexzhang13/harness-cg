#!/usr/bin/env bash
# Train one method on one task pair of the strategy-generalization experiment (Fig. strategy-gen).
#
#   [BACKEND=tinker] bash run_train.sh <method> <task> [extra `rl` args]
#     method: rlm | lm_repl | pi | transformer
#     task:   oolong_trec_to_spam | obliq_writing_to_math | obliq_twitter_to_wildchat
#
# Evaluated inline on the held-out domain. RLM and LM + REPL configs include a task hint.
set -euo pipefail
METHOD=${1:?method}; TASK=${2:?task}; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

if [[ "$METHOD" == pi ]]; then
  export PATH="${PI_NODE_BIN:-$HOME/node/bin}:$PATH"
  export VF_LOCAL_SANDBOX_NODE_BIN="${VF_LOCAL_SANDBOX_NODE_BIN:-${PI_NODE_BIN:-$HOME/node/bin}}"
  # Pi hard caps (turns, sub-agent budget, wall clock), shared with the length-generalization Pi runs
  eval "$(python3 "$REPO/experiments/sec3.1_length_generalization/pi_runs/settings.py" env)"
fi
# 300 s per sub-call for the OBLIQ RLM runs
if [[ "$METHOD" == rlm && "$TASK" == obliq_* ]]; then
  export RLM_TRAIN_SUBCALL_TIMEOUT_S="${RLM_TRAIN_SUBCALL_TIMEOUT_S:-300}"
fi
LAUNCH="$REPO/training/prime_rl/launch.sh"
[[ "${BACKEND:-prime-rl}" == tinker ]] && LAUNCH="$REPO/training/tinker/launch.sh"
exec bash "$LAUNCH" "$HERE/configs/$METHOD/$TASK.toml" "$@"
