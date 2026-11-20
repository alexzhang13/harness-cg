#!/usr/bin/env bash
# Long-split evaluation of the base Transformer's saved adapters (Fig. length-gen, "Base Transformer + YaRN").
#
#   bash eval_transformer.sh <task>      task: mrcr | graphwalks | longbenchpro | oolong | oolong-pairs
#
# Scores every saved adapter at temperature 0, with YaRN where the split exceeds 262144 tokens. Ada-LEval is
# evaluated inline during training.
set -euo pipefail
TASK=${1:?task}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
RUN="$REPO/training/prime_rl/offline_eval/run_offline_lora.sh"
OUTPUTS="$PRIME_RL_DIR/outputs/length-gen/transformer"

case "$TASK" in
  mrcr)
    # 1m-2m split (2m-4m does not fit).
    YARN=1 TP=8 GPU_UTIL=0.90 CHUNK=131072 MAX_NUM_SEQS=1 MAXCONC=1 EVAL_BASE=0 \
      EXTRA_VLLM_ARGS="--disable-custom-all-reduce" \
      bash "$RUN" "$OUTPUTS/mrcr" mrcr \
      '{"needle_count":8,"context_range":"1m-2m","seed":42}' 2129920 50 4096 ;;
  graphwalks)
    YARN=1 CHUNK=16384 MAXCONC=1 GPU_UTIL=0.92 \
      bash "$RUN" "$OUTPUTS/graphwalks" graphwalks \
      '{"split":"256k_to_1mil","seed":42}' 1048576 50 4096 ;;
  longbenchpro)
    CHUNK=16384 MAXCONC=1 GPU_UTIL=0.92 \
      bash "$RUN" "$OUTPUTS/longbenchpro" longbenchpro \
      '{"language":"English","token_length":"256k","seed":42}' 262144 115 8192 ;;
  oolong)
    # answer-only prompt; otherwise the model spends its output budget reasoning
    SP='Respond with ONLY the final answer, in the exact format requested by the question. Output nothing else: no reasoning, no explanation, no restating the question, no listing the data. Emit only the single answer line.'
    EVAL_BASE=1 RESUME=0 EAGER=0 MAX_NUM_SEQS=8 \
      OUT="$OUTPUTS/oolong/offline_long_eval_lora_answeronly.jsonl" \
      bash "$RUN" "$OUTPUTS/oolong" oolong \
      "{\"dataset_name\":\"trec_coarse\",\"min_ctx\":262144,\"max_ctx\":262144,\"seed\":42,\"system_prompt\":\"$SP\"}" \
      262144 50 8192 ;;
  oolong-pairs)
    bash "$RUN" "$OUTPUTS/oolong-pairs" oolong_pairs \
      '{"context_length":32768,"num_examples":20,"seed":42}' 196608 20 131072 ;;
  *) echo "unknown task: $TASK" >&2; exit 1 ;;
esac
