#!/usr/bin/env bash
# Broad mixture (Fig. broad-mixture-rl): RL of Qwen3-30B-A3B on 26 streams, 18 held-out validation streams.
# The paper evaluates the step-480 adapter on the held-out suite.
#
#   [BACKEND=tinker] bash run_train.sh [extra `rl` args]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH="$HERE/../../../training/prime_rl/launch.sh"
[[ "${BACKEND:-prime-rl}" == tinker ]] && LAUNCH="$HERE/../../../training/tinker/launch.sh"
exec bash "$LAUNCH" "$HERE/configs/train-mix-large.toml" "$@"
