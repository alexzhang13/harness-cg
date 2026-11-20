#!/usr/bin/env bash
# Train one method on the mixture of the six short length-generalization tasks (Table generalization).
#
#   [BACKEND=tinker] bash run_train.sh <rlm|lm_repl|pi|transformer> [extra `rl` args]
#
# The single-task rows are the length-gen runs:
#   bash ../../sec3.1_length_generalization/run_train.sh rlm_orchestrator_prompt <graphwalks|longbenchpro|oolong-pairs>
set -euo pipefail
METHOD=${1:?method}; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
export ADALEVAL_DATA_DIR="${ADALEVAL_DATA_DIR:-$HOME/ada_leval_data}"
if [[ "$METHOD" == pi ]]; then
  export PATH="${PI_NODE_BIN:-$HOME/node/bin}:$PATH"
  export VF_LOCAL_SANDBOX_NODE_BIN="${VF_LOCAL_SANDBOX_NODE_BIN:-${PI_NODE_BIN:-$HOME/node/bin}}"
fi
LAUNCH="$REPO/training/prime_rl/launch.sh"
[[ "${BACKEND:-prime-rl}" == tinker ]] && LAUNCH="$REPO/training/tinker/launch.sh"
exec bash "$LAUNCH" "$HERE/configs/$METHOD.toml" "$@"
