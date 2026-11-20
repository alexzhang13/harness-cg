#!/usr/bin/env bash
# Runs one pipeline work item on this host (started by the orchestrator; also usable by hand):
#   bash synth-datagen/common/run_local.sh rlm_generate_trajectories.pipeline.worker <pipeline dir>/work/<id>.json
# Judge credentials: PRIME_API_KEY or RLM_JUDGE_*. OPENAI_* is unset; the teacher comes from the spec.
set -euo pipefail
MOD=$1; shift
RLM_TRAIN_DIR="${RLM_TRAIN_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
PP="$RLM_TRAIN_DIR/src:$RLM_TRAIN_DIR/synth-datagen/common"
for d in "$RLM_TRAIN_DIR"/environments/real/*/ "$RLM_TRAIN_DIR"/environments/synthetic/*/; do
  [[ -f "$d/pyproject.toml" ]] && PP="$PP:${d%/}"
done
cd "$RLM_TRAIN_DIR"
unset OPENAI_BASE_URL OPENAI_API_KEY
export PYTHONPATH="$PP" HF_DATASETS_IN_MEMORY_MAX_SIZE=0 TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export RLM_TRAIN_CODE_BLOCK_FEEDBACK=${RLM_TRAIN_CODE_BLOCK_FEEDBACK:-1}  # Section 4 setting: feedback on turns without a code block
export RLM_TRAIN_EXEC_TIMEOUT_S=${RLM_TRAIN_EXEC_TIMEOUT_S:-600} RLM_TRAIN_SUBCALL_TIMEOUT_S=${RLM_TRAIN_SUBCALL_TIMEOUT_S:-600}
export RLM_JUDGE_MODEL=${RLM_JUDGE_MODEL:-openai/gpt-5-mini}
exec nice -n 10 "${PIPELINE_PYTHON:-python}" -m "$MOD" "$@"
