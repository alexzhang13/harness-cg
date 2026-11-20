#!/usr/bin/env bash
# Serve one model with vLLM on the visible GPUs (one server per TP-sized GPU group) and register each server in a
# pool file for lb_proxy.py.
#
#   MODEL=<HF id or local dir> SERVED=<name> POOL=pool.json bash serve_vllm.sh
#
# Also: TP (2), MAX_MODEL_LEN (262144), GPU_UTIL, MAX_NUM_SEQS, PORT_BASE, TOKENIZER, LORA="name=path ...", YARN=1,
# REASONING_PARSER (qwen3 for SFT trace generation, unset for evals), EXTRA_VLLM_ARGS.
set -uo pipefail
MODEL="${MODEL:?}"; SERVED="${SERVED:?}"; POOL="${POOL:?}"
TP="${TP:-2}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"; GPU_UTIL="${GPU_UTIL:-0.92}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PORT_BASE="${PORT_BASE:-8800}"; TOKENIZER="${TOKENIZER:-$MODEL}"
VLLM="${VLLM:-vllm}"
GL="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}"
IFS=',' read -ra G <<< "$GL"; NSERV=$(( ${#G[@]} / TP ))
IP=$(hostname -I | awk '{print $1}')
[[ -f "$POOL" ]] || echo '{"ports": []}' > "$POOL"

EXTRA=(--tokenizer "$TOKENIZER")
[[ -n "${REASONING_PARSER:-}" ]] && EXTRA+=(--reasoning-parser "$REASONING_PARSER")
[[ -n "${LORA:-}" ]] && EXTRA+=(--enable-lora --max-lora-rank 32 --max-loras 4 --lora-modules $LORA)
if [[ "${YARN:-0}" == 1 ]]; then
  factor=$(python3 -c "print(max(1.0, $MAX_MODEL_LEN/262144))")
  EXTRA+=(--hf-overrides "{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":$factor,\"original_max_position_embeddings\":262144}}")
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi

pool_edit() {  # add|del <url>
  python3 - "$POOL" "$1" "$2" "$MAX_MODEL_LEN" <<'PY'
import fcntl, json, os, sys
f, op, url, ctx = sys.argv[1:5]
with open(f + ".lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    d = json.load(open(f)); d["ports"] = [p for p in d["ports"] if p["url"] != url]
    if op == "add":
        d["ports"].append({"url": url, "ctx": int(ctx)})
    json.dump(d, open(f + ".tmp", "w")); os.replace(f + ".tmp", f)
PY
}

PIDS=()
for i in $(seq 0 $((NSERV - 1))); do
  gpus=$(IFS=,; echo "${G[*]:$((i * TP)):$TP}"); port=$((PORT_BASE + i))
  CUDA_VISIBLE_DEVICES=$gpus VLLM_USE_FLASHINFER_SAMPLER=0 setsid "$VLLM" serve "$MODEL" --served-model-name "$SERVED" \
    --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" --host 0.0.0.0 --port "$port" --trust-remote-code "${EXTRA[@]}" ${EXTRA_VLLM_ARGS:-} \
    > "vllm-${SERVED//\//_}-$port.log" 2>&1 &
  PIDS+=($!)
done
trap 'for i in "${!PIDS[@]}"; do pool_edit del "http://$IP:$((PORT_BASE + i))/v1"; kill -TERM -- -"${PIDS[$i]}" 2>/dev/null; done' EXIT
for i in "${!PIDS[@]}"; do
  port=$((PORT_BASE + i))
  for _ in $(seq 1 240); do curl -sf -m 10 "http://localhost:$port/health" >/dev/null && break; sleep 10; done
  curl -sf -m 10 "http://localhost:$port/health" >/dev/null && pool_edit add "http://$IP:$port/v1" && echo "server $i ready on :$port"
done
wait
