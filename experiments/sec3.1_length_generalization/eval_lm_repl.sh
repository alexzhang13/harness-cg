#!/usr/bin/env bash
# Long-split evaluation of LM + REPL's saved adapters (configs that do not evaluate inline).
#
#   bash eval_lm_repl.sh <task>      task: oolong
#
# 262,144-token window, sampling from the model's generation_config as in inline evals.
# Writes <run>/offline_long_eval_lora.jsonl.
set -euo pipefail
TASK=${1:?task}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
RUN="$REPO/training/prime_rl/offline_eval/run_offline_lora.sh"
OUTPUTS="$PRIME_RL_DIR/outputs/length-gen/lm_repl"

case "$TASK" in
  oolong)
    ENV_ID=oolong-real TEMP=-1 EAGER=0 CHUNK=32768 MAX_NUM_SEQS=16 MAXCONC=16 \
      bash "$RUN" "$OUTPUTS/oolong" oolong \
      '{"dataset_name":"trec_coarse","min_ctx":262144,"max_ctx":262144,"exclude_numeric":false,"seed":42,"max_iterations":20,"sub_max_tokens":4096,"enable_sub_llm":false,"code_block_feedback":true}' \
      262144 50 4096 ;;
  *) echo "unknown task: $TASK" >&2; exit 1 ;;
esac
