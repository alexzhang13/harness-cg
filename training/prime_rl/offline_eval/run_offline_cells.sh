#!/usr/bin/env bash
# Serve the base model with a run's adapters and evaluate every cell for every checkpoint at temperature 0.
#
#   RUN_DIR=<run dir> STEPS=1000 CELLS=cells.jsonl OUT=results.jsonl MAX_MODEL_LEN=131072 run_offline_cells.sh
#
# CELLS is JSONL, one cell per line:
#   {"label": "...", "env_id": "<verifiers env>" | "task": "<singleturn-baseline task>",
#    "args": {...}, "num_examples": N, "maxtok": optional per-cell output cap}
# One summary row per (cell, checkpoint) is appended to OUT (see offline_long_eval.py).
#
# Also: EVAL_BASE=1 (score step 0), MAXTOK (8192), MAXCONC, REQ_TIMEOUT, YARN=1, EAGER, MAX_NUM_SEQS, CHUNK, TP,
# GPU_UTIL, PORT, EXTRA_VLLM_ARGS, VENV_BIN.
set -uo pipefail
RUN_DIR=${RUN_DIR:?}; STEPS=${STEPS:?comma list}; CELLS=${CELLS:?}; OUT=${OUT:?}; MAX_MODEL_LEN=${MAX_MODEL_LEN:?}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIME_RL_DIR=${PRIME_RL_DIR:-$HOME/prime-rl}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
VENV=${VENV_BIN:-$PRIME_RL_DIR/.venv/bin}; TP=${TP:-4}; PORT=${PORT:-8011}; GPU_UTIL=${GPU_UTIL:-0.90}
EVAL_BASE=${EVAL_BASE:-0}; MAXTOK=${MAXTOK:-8192}
export PATH="$VENV:$PATH" TRITON_CACHE_DIR="/tmp/triton_off_$$_$PORT"; mkdir -p "$TRITON_CACHE_DIR" "$(dirname "$OUT")"

LORA=(); STEP_LIST=()
for s in ${STEPS//,/ }; do
  d="$RUN_DIR/run_default/broadcasts/step_$s"
  [[ -f $d/adapter_model.safetensors ]] || { echo "no adapter $d"; exit 1; }
  LORA+=("step_$s=$d"); STEP_LIST+=("$s")
done

YARN_ARGS=(); YARN_ENV=()
if [[ "${YARN:-0}" == "1" ]]; then
  factor=$(python3 -c "print(max(1.0, $MAX_MODEL_LEN/262144))")
  rope="{\"rope_type\":\"yarn\",\"factor\":$factor,\"original_max_position_embeddings\":262144,\"rope_theta\":10000000}"
  YARN_ARGS=(--hf-overrides "{\"rope_scaling\":$rope,\"rope_parameters\":$rope}")
  YARN_ENV=(VLLM_ALLOW_LONG_MAX_MODEL_LEN=1)
fi
FIT=(); [[ "${EAGER:-0}" == "1" ]] && FIT+=(--enforce-eager); FIT+=(--max-num-seqs "${MAX_NUM_SEQS:-4}")
[[ -n "${CHUNK:-}" ]] && FIT+=(--enable-chunked-prefill --max-num-batched-tokens "$CHUNK")

VLOG=$(mktemp /tmp/vllm_cells_XXXX.log)
setsid env VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 "${YARN_ENV[@]}" \
  "$VENV/vllm" serve "$BASE_MODEL" --served-model-name base --enable-lora --max-lora-rank 32 --max-loras 1 \
  --max-cpu-loras "$(( ${#STEP_LIST[@]} + 1 ))" --lora-modules "${LORA[@]}" \
  --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" --port "$PORT" \
  --trust-remote-code "${FIT[@]}" "${YARN_ARGS[@]}" ${EXTRA_VLLM_ARGS:-} >"$VLOG" 2>&1 &
VPID=$!
trap 'kill -TERM -- -$VPID 2>/dev/null; sleep 20; kill -KILL -- -$VPID 2>/dev/null; wait $VPID 2>/dev/null' EXIT
ok=0
for _ in $(seq 1 240); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $VPID 2>/dev/null || break
  sleep 10
done
[[ $ok == 1 ]] || { echo "server failed:"; tail -60 "$VLOG"; exit 1; }

MODELS=(); for s in "${STEP_LIST[@]}"; do MODELS+=("step_$s:$s"); done; [[ $EVAL_BASE == 1 ]] && MODELS+=("base:0")
field() { python3 -c 'import sys,json; d=json.loads(sys.argv[1]); v=d.get(sys.argv[2], ""); print(json.dumps(v) if isinstance(v,(dict,list)) else v)' "$1" "$2"; }
while IFS= read -r line <&3; do
  [[ -z "$line" ]] && continue
  label=$(field "$line" label); env_id=$(field "$line" env_id); task=$(field "$line" task)
  args=$(field "$line" args); n=$(field "$line" num_examples); mt=$(field "$line" maxtok)
  target=(--task "$task"); [[ -n "$env_id" ]] && target=(--env-id "$env_id")
  for m in "${MODELS[@]}"; do
    echo "=== $label  model=${m%%:*} ==="
    "$VENV/python" "$HERE/offline_long_eval.py" "${target[@]}" --label "$label" --args-json "${args:-{\}}" \
      --num-examples "$n" --base-url "http://localhost:$PORT/v1" --model "${m%%:*}" \
      --max-completion-tokens "${mt:-$MAXTOK}" --temperature 0.0 --step "${m##*:}" --out "$OUT" \
      --max-concurrent "${MAXCONC:-8}" --request-timeout "${REQ_TIMEOUT:-900}" --chunk-size "${EVAL_CHUNK:-0}" \
      ${SHARD:+--shard "$SHARD"} </dev/null || echo "FAILED $label ${m%%:*}"
  done
done 3< "$CELLS"
echo "done -> $OUT"
