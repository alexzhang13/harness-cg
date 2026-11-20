#!/usr/bin/env bash
# Serve the base model with every LoRA adapter of a run and score step 0 and each adapter on one env's eval split.
#
#   run_offline_lora.sh <run_dir> <task> '<env args JSON>' [max_model_len=262144] [num_examples=50] [max_tokens=4096]
#
# Adapters: <run_dir>/run_default/broadcasts/step_N. <task> is a singleturn-baseline task (or a label, with ENV_ID).
# One row per step is appended to $OUT (default <run_dir>/offline_long_eval_lora.jsonl).
#
# Environment:
#   YARN=1            YaRN, factor max(1, max_model_len / 262144)
#   STEPS=10,20       only these steps; RESUME=1 (default) skips steps with a clean row in $OUT
#   EVAL_BASE=0       skip the base model (step 0)
#   TEMP              temperature (default 0); negative = the model's generation_config, as inline evals
#   TP, GPU_UTIL, PORT, MAXCONC            tensor parallel size (8), GPU memory fraction (0.90), port, client concurrency (8)
#   CHUNK, MAX_NUM_SEQS, EAGER, FORCE_FIT  long-context settings (used at >= 262144 tokens, YARN=1 or FORCE_FIT=1)
#   EXTRA_VLLM_ARGS   appended to `vllm serve`
#   VENV_BIN          directory holding `python` and `vllm` (default $PRIME_RL_DIR/.venv/bin)
set -euo pipefail

RUN_DIR=${1:?run dir}
TASK=${2:?task}
ARGS_JSON=${3:?env args JSON}
MAX_MODEL_LEN=${4:-262144}
NUM_EX=${5:-50}
MAXTOK=${6:-4096}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIME_RL_DIR=${PRIME_RL_DIR:-$HOME/prime-rl}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
TP=${TP:-8}
PORT=${PORT:-8011}
GPU_UTIL=${GPU_UTIL:-0.90}
VENV="${VENV_BIN:-$PRIME_RL_DIR/.venv/bin}"
OUT="${OUT:-$RUN_DIR/offline_long_eval_lora.jsonl}"
BCAST="$RUN_DIR/run_default/broadcasts"
export PATH="$VENV:$PATH"
export TRITON_CACHE_DIR="/tmp/triton_off_$$"
mkdir -p "$TRITON_CACHE_DIR"

shopt -s nullglob
LORA_ARGS=(); STEP_LIST=()
for d in $(ls -d "$BCAST"/step_* 2>/dev/null | sort -t_ -k2 -n); do
  step=$(basename "$d" | sed 's/step_//')
  [[ -f "$d/adapter_model.safetensors" ]] || continue
  if [[ -n "${STEPS:-}" ]] && ! echo " ${STEPS//,/ } " | grep -q " $step "; then continue; fi
  LORA_ARGS+=("step_$step=$d")
  STEP_LIST+=("$step")
done
EVAL_BASE=${EVAL_BASE:-1}
[[ ${#STEP_LIST[@]} -gt 0 || "$EVAL_BASE" == "1" ]] || { echo "no adapters under $BCAST and EVAL_BASE=0"; exit 1; }
[[ -z "${STEPS:-}" ]] && : > "$OUT"

# resume: skip steps that already have a clean row
if [[ -n "${STEPS:-}" && "${RESUME:-1}" == "1" && -f "$OUT" ]]; then
  GOOD=$("$VENV/python" - "$OUT" <<'PY'
import json, sys
good = set()
for line in open(sys.argv[1]):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("n_error", 0) == 0 and r.get("n", 0) > 0:
        good.add(int(r["step"]))
print(",".join(map(str, sorted(good))))
PY
)
  if [[ -n "$GOOD" ]]; then
    NEW_LORA=(); NEW_STEPS=()
    for i in "${!STEP_LIST[@]}"; do
      s="${STEP_LIST[$i]}"
      echo " ${GOOD//,/ } " | grep -q " $s " && continue
      NEW_STEPS+=("$s"); NEW_LORA+=("${LORA_ARGS[$i]}")
    done
    STEP_LIST=("${NEW_STEPS[@]}"); LORA_ARGS=("${NEW_LORA[@]}")
    if [[ "$EVAL_BASE" == "1" ]] && echo " ${GOOD//,/ } " | grep -q " 0 "; then EVAL_BASE=0; fi
    [[ ${#STEP_LIST[@]} -gt 0 || "$EVAL_BASE" == "1" ]] || { echo "nothing left to evaluate"; exit 0; }
  fi
fi

YARN_ARGS=(); YARN_ENV=()
if [[ "${YARN:-0}" == "1" ]]; then
  factor=$(python3 -c "print(max(1.0, $MAX_MODEL_LEN/262144))")
  YARN_ARGS=(--hf-overrides "{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":$factor,\"original_max_position_embeddings\":262144}}")
  YARN_ENV=(VLLM_ALLOW_LONG_MAX_MODEL_LEN=1)
fi

FIT_ARGS=()
if [[ "$MAX_MODEL_LEN" -ge 262144 || "${YARN:-0}" == "1" || "${FORCE_FIT:-0}" == "1" ]]; then
  EAGER_ARG=(); [[ "${EAGER:-1}" == "1" ]] && EAGER_ARG=(--enforce-eager)
  FIT_ARGS=(--enable-chunked-prefill --max-num-batched-tokens "${CHUNK:-131072}" "${EAGER_ARG[@]}" --max-num-seqs "${MAX_NUM_SEQS:-1}")
fi

LORA_SERVE_ARGS=()
if [[ ${#STEP_LIST[@]} -gt 0 ]]; then
  LORA_SERVE_ARGS=(--enable-lora --max-lora-rank 32 --max-loras 1
                   --max-cpu-loras "$(( ${#STEP_LIST[@]} + 1 ))"
                   --lora-modules "${LORA_ARGS[@]}")
fi

VLLM_LOG=$(mktemp)
env VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 "${YARN_ENV[@]}" \
  "$VENV/vllm" serve "$BASE_MODEL" \
    --served-model-name base \
    "${LORA_SERVE_ARGS[@]}" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" --trust-remote-code "${FIT_ARGS[@]}" "${YARN_ARGS[@]}" ${EXTRA_VLLM_ARGS:-} >"$VLLM_LOG" 2>&1 &
VPID=$!
trap 'kill "$VPID" 2>/dev/null || true; wait "$VPID" 2>/dev/null || true' EXIT
ok=0
for _ in $(seq 1 180); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
  kill -0 "$VPID" 2>/dev/null || { echo "vLLM died:"; tail -200 "$VLLM_LOG"; break; }
  sleep 10
done
[[ "$ok" == 1 ]] || { echo "server never became healthy"; exit 1; }

eval_one() {  # <served model> <step>
  local task_args=(--task "$TASK")
  [[ -n "${ENV_ID:-}" ]] && task_args=(--env-id "$ENV_ID" --label "$TASK")
  "$VENV/python" "$HERE/offline_long_eval.py" \
    "${task_args[@]}" --args-json "$ARGS_JSON" --num-examples "$NUM_EX" \
    --base-url "http://localhost:$PORT/v1" --model "$1" \
    --max-completion-tokens "$MAXTOK" --temperature "${TEMP:-0.0}" --step "$2" --out "$OUT" \
    --max-concurrent "${MAXCONC:-8}" \
    || echo "eval of $1 FAILED"
}
[[ "$EVAL_BASE" == "1" ]] && eval_one base 0
for step in "${STEP_LIST[@]}"; do
  eval_one "step_$step" "$step"
done
echo "curve -> $OUT"
