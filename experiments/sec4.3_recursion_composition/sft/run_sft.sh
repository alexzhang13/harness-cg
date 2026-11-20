#!/usr/bin/env bash
# "+ SFT" rows of Table recursion-composition: SFT of RLM-27B or Qwen3.8-27B on the gated fake-task corpus.
#
#   PRIME_RL_SFT_GATED_DIR=<checkout with patches/sft_gated> SFT_DATA_DIR=<dir with train/validation.parquet> \
#     [INIT_WEIGHTS=<RLM-27B dir>] bash sft/run_sft.sh <rlm-27b|qwen38-27b>
#
# 300-step cosine schedule; the paper uses step 100, so training stops once weights/step_100 is written.
set -euo pipefail
WHICH=${1:?rlm-27b|qwen38-27b}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO="$(cd "$HERE/../../.." && pwd)"
D="${PRIME_RL_SFT_GATED_DIR:?}"; : "${SFT_DATA_DIR:?}"
[[ "$WHICH" == rlm-27b ]] && : "${INIT_WEIGHTS:?set INIT_WEIGHTS to a local copy of a1zhang/RLM-27B}"
STOP_AFTER="${STOP_AFTER:-100}"
CONFIG="$(mktemp --suffix=.toml)"
envsubst '${REPO} ${SFT_DATA_DIR} ${INIT_WEIGHTS}' < "$HERE/$WHICH.toml" > "$CONFIG"
OUT="$D/$(grep -aoE '^output_dir = "[^"]+"' "$CONFIG" | sed -E 's/.*"([^"]+)".*/\1/')"
[[ -f "$D/.env" ]] && { set -a; source "$D/.env"; set +a; }
cd "$D"
mkdir -p "$(dirname "$OUT")"
setsid uv run sft @ "$CONFIG" > "$OUT.train.log" 2>&1 &
TPID=$!
until [[ -f "$OUT/weights/step_$STOP_AFTER/STABLE" ]] || ! kill -0 $TPID 2>/dev/null; do sleep 30; done
[[ -f "$OUT/weights/step_$STOP_AFTER/STABLE" ]] || { echo "trainer exited before step $STOP_AFTER; see $OUT.train.log"; exit 1; }
kill -TERM -- -$TPID 2>/dev/null || true; sleep 20
python "$REPO/training/sft/fix_checkpoint.py" --base Qwen/Qwen3.8-27B "$OUT/weights/step_$STOP_AFTER"
echo "servable checkpoint: $OUT/weights/step_$STOP_AFTER"
