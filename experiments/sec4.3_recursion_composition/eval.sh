#!/usr/bin/env bash
# Evaluate one cell of Table recursion-composition on the 2,000 gated chains (data/chains2000.json).
#
#   BASE_URL=http://127.0.0.1:18431/v1 MODEL=<served name> bash eval.sh <harness|hint|sft> [extra run_gated.py args]
#
#   harness  gflat: unlock only, protocol instruction before the question, no progress line, 80 turns
#   hint     ghint: unlock + handoff, required-protocol hint, no progress line, 60 turns
#   sft      gplain: unlock + handoff, no instruction, progress line, 20 turns (for the SFT checkpoints)
# All cells: temperature 0, thinking sub-calls (24576 tokens), 2 attempts per task.
# --chains-file data/chains_dev20.json for a quick check.
set -euo pipefail
CELL=${1:?cell}; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${BASE_URL:?}" "${MODEL:?}"
OUT="${OUT:-$HERE/runs/$MODEL-$CELL}"
case "$CELL" in
  harness) ARGS=(--conds gflat --max-iterations 80 --timeout 3600) ;;
  hint)    ARGS=(--conds ghint --max-iterations 60 --timeout 3600) ;;
  sft)     ARGS=(--conds gplain --header post --max-iterations 20) ;;
  *) echo "unknown cell: $CELL" >&2; exit 1 ;;
esac
exec python "$HERE/run_gated.py" --base-url "$BASE_URL" --model "$MODEL" --chains-file "$HERE/data/chains2000.json" \
  --temperature 0 --sub-thinking --save-traj --resume --out "$OUT" "${ARGS[@]}" "$@"
