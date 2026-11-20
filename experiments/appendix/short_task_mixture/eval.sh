#!/usr/bin/env bash
# Evaluate a short-mixture checkpoint on the nine cells of Table generalization, temperature 0.
#
#   bash eval.sh <rlm|lm_repl|pi|transformer> [group]
#
# Paper checkpoints: rlm 1000, lm_repl 1000, pi 300, transformer 420; override with STEPS=.
# EVAL_BASE=1 also scores the base model. The Transformer runs in groups, one server window each:
#   A  188416 window:        OOLONG-Pairs, Ada-LEval, OOLONG spam, OBLIQ math, OBLIQ wildchat
#   B  270336 window, YaRN:  OOLONG 256k, LongBench-Pro 256k
#   B2 294912 window, YaRN:  LongBench-Pro 256k (prompts that exceed the B window)
#   E  262144 window:        OOLONG-Pairs with a 230000-token output cap
#   C2 1474560 window, YaRN: GraphWalks
#   D2 2129920 window, YaRN, fp8 KV cache: MRCRv2 at 1m-2m (the 2m-4m split does not fit)
set -euo pipefail
METHOD=${1:?method}; GROUP=${2:-all}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
RUN="$REPO/training/prime_rl/offline_eval/run_offline_cells.sh"
export RUN_DIR="$PRIME_RL_DIR/outputs/short-mix/$METHOD"
export ADALEVAL_DATA_DIR="${ADALEVAL_DATA_DIR:-$HOME/ada_leval_data}"
OUTDIR="$RUN_DIR/offline_eval"

case "$METHOD" in
  rlm|lm_repl)
    STEPS="${STEPS:-1000}" TP=8 MAX_MODEL_LEN=131072 MAXTOK=4096 MAXCONC=12 \
      CELLS="$HERE/cells/$METHOD.jsonl" OUT="$OUTDIR/cells.jsonl" bash "$RUN" ;;
  pi)
    export PATH="${PI_NODE_BIN:-$HOME/node/bin}:$PATH"
    export VF_LOCAL_SANDBOX_NODE_BIN="${VF_LOCAL_SANDBOX_NODE_BIN:-${PI_NODE_BIN:-$HOME/node/bin}}"
    export PI_SUBAGENT_MAX_TURNS=10 PI_CONTEXT_WINDOW=131072 PI_COMPACT_AT_TOKENS=65536 PI_COMPACT_KEEP_RECENT_TOKENS=8192
    STEPS="${STEPS:-300}" TP=8 MAX_MODEL_LEN=131072 MAXTOK=8192 MAXCONC=8 \
      EXTRA_VLLM_ARGS="--enable-auto-tool-choice --tool-call-parser hermes" \
      CELLS="$HERE/cells/pi.jsonl" OUT="$OUTDIR/cells.jsonl" bash "$RUN" ;;
  transformer)
    export STEPS="${STEPS:-420}" TP=4 MAXTOK=8192 REQ_TIMEOUT=14400 EVAL_BASE="${EVAL_BASE:-1}"
    export EXTRA_VLLM_ARGS="--disable-custom-all-reduce"
    group() { [[ "$GROUP" == all || "$GROUP" == "$1" ]]; }
    cells() { echo "$HERE/cells/transformer_$1.jsonl"; }
    group A  && CELLS="$(cells A_native_180k)" OUT="$OUTDIR/A_native.jsonl" MAX_MODEL_LEN=188416 MAX_NUM_SEQS=8 bash "$RUN"
    group B  && CELLS="$(cells B_256k_yarn)" OUT="$OUTDIR/B_256k.jsonl" MAX_MODEL_LEN=270336 YARN=1 MAX_NUM_SEQS=2 bash "$RUN"
    group B2 && CELLS="$(cells B2_lbp_yarn)" OUT="$OUTDIR/B2_lbp.jsonl" MAX_MODEL_LEN=294912 YARN=1 MAX_NUM_SEQS=2 EVAL_CHUNK=23 bash "$RUN"
    group E  && CELLS="$(cells E_pairs_longout)" OUT="$OUTDIR/E_pairs_longout.jsonl" MAX_MODEL_LEN=262144 MAX_NUM_SEQS=8 bash "$RUN"
    group C2 && CELLS="$(cells C2_graphwalks_1p5m)" OUT="$OUTDIR/C2_graphwalks.jsonl" MAX_MODEL_LEN=1474560 YARN=1 \
      MAX_NUM_SEQS=1 EAGER=1 CHUNK=131072 GPU_UTIL=0.92 EVAL_CHUNK=5 bash "$RUN"
    group D2 && CELLS="$(cells D2_mrcr_fp8kv)" OUT="$OUTDIR/D2_mrcr.jsonl" MAX_MODEL_LEN=2129920 YARN=1 \
      MAX_NUM_SEQS=1 EAGER=1 CHUNK=131072 GPU_UTIL=0.92 EVAL_CHUNK=5 \
      EXTRA_VLLM_ARGS="--disable-custom-all-reduce --kv-cache-dtype fp8" bash "$RUN"
    true ;;
  *) echo "unknown method: $METHOD" >&2; exit 1 ;;
esac
