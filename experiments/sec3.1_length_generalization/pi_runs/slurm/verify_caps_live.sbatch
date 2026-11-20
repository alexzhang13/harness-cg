#!/usr/bin/env bash
#SBATCH -J cgpi-verify-caps
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -t 01:00:00
# Serve the base model on 2 GPUs and run verify_caps_live.py against it.
#   sbatch --export=ALL,PI_RUNS_DIR=$PWD/.., OUT=/path/to/out [-p <partition>] verify_caps_live.sbatch
set -uo pipefail
PI_RUNS="${PI_RUNS_DIR:?set PI_RUNS_DIR to the pi_runs directory}"
OUT="${OUT:?set OUT}"
source "$PI_RUNS/env.sh" || exit 1
unset PI_PROMPT_VARIANT   # the probe must not be told about the limits
PORT=$((8700 + ${SLURM_JOB_ID:-0} % 200))
mkdir -p "$OUT"
setsid "$LOCAL_ENV/bin/vllm" serve Qwen/Qwen3-30B-A3B-Instruct-2507 --served-model-name base \
  --tensor-parallel-size 2 --max-model-len 65536 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes \
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}' --port "$PORT" --trust-remote-code >"$OUT/vllm.log" 2>&1 &
VPID=$!
trap 'kill -- -"$VPID" 2>/dev/null; sleep 5; kill -9 -- -"$VPID" 2>/dev/null' EXIT
for _ in $(seq 1 240); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$VPID" 2>/dev/null || { echo "vLLM died"; tail -40 "$OUT/vllm.log"; exit 1; }
  sleep 10
done
cd "$PI_RUNS"
"$PRIME_RL_PYTHON" verify_caps_live.py --base-url "http://localhost:$PORT/v1" --model base --out "$OUT" ${ONLY:+--only $ONLY}
