#!/usr/bin/env bash
#SBATCH -J cgpi-eval
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH -t 1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
# 2-GPU vLLM server with runtime LoRA loading plus the eval driver in --watch mode; several jobs can share a run.
#   sbatch -J cgpi-eval-<task>-s<seed> -o <RUN>/eval/logs/slurm-%j.out \
#          --export=ALL,RUN=<run dir>,TASK=<mrcr|longbenchpro|oolong-pairs|adaleval> slurm/eval_server.sbatch
# Also: EXIT_AFTER_IDLE (3600 s), EXTRA_DRIVER_ARGS.
set -uo pipefail
PI_RUNS="${PI_RUNS_DIR:?set PI_RUNS_DIR to the pi_runs directory}"
RUN="${RUN:?set RUN=<run dir>}"; TASK="${TASK:?set TASK=mrcr|longbenchpro|oolong-pairs|adaleval}"
OUT="${OUT:-$RUN/eval}"
mkdir -p "$OUT/logs" "$OUT/vllm_logs"
source "$PI_RUNS/env.sh" || exit 1

# Training limits from env.sh, except longer wall clocks and the full 262144 window. Each shard manifest records the
# PI_* values.
export PI_WALL_TIMEOUT_S=1800 PI_SUBAGENT_WALL_TIMEOUT_S=300 PI_COMMAND_TIMEOUT=1860 PI_ROLLOUT_TIMEOUT=1920
# per-task overrides (settings.TASK_EVAL_ENV)
eval "$("$PRIME_RL_PYTHON" "$PI_RUNS/settings.py" eval-env "$TASK")"
export PI_CONTEXT_WINDOW=262144
unset PI_READCAP_DISABLE PI_COMPACT_AT_TOKENS PI_COMPACT_KEEP_RECENT_TOKENS

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 TOKENIZERS_PARALLELISM=false OPENAI_API_KEY=x
export TRITON_CACHE_DIR="/tmp/triton_${USER}_eval_${SLURM_JOB_ID:-$$}"; mkdir -p "$TRITON_CACHE_DIR"
PORT=$((8700 + ${SLURM_JOB_ID:-0} % 200))
VLOG="$OUT/vllm_logs/vllm-${SLURM_JOB_ID:-local}-r${SLURM_RESTART_COUNT:-0}.log"
setsid "$LOCAL_ENV/bin/vllm" serve Qwen/Qwen3-30B-A3B-Instruct-2507 --served-model-name base \
  --enable-lora --max-lora-rank 32 --max-loras 4 --max-cpu-loras 24 \
  --tensor-parallel-size 2 --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill --max-num-batched-tokens 16384 --max-num-seqs 96 \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes \
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}' --port "$PORT" --trust-remote-code >"$VLOG" 2>&1 &
VPID=$!
trap 'kill -- -"$VPID" 2>/dev/null; sleep 10; kill -9 -- -"$VPID" 2>/dev/null' EXIT
for _ in $(seq 1 240); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$VPID" 2>/dev/null || { echo "[eval-server] vLLM died"; tail -60 "$VLOG"; exit 1; }
  sleep 10
done
curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "[eval-server] vLLM never healthy"; tail -60 "$VLOG"; exit 1; }
echo "[eval-server] vLLM up on :$PORT ($(hostname)) job=${SLURM_JOB_ID:-local} run=$RUN task=$TASK"

cd "$HCG/experiments/sec3.1_length_generalization"
# shellcheck disable=SC2086
"$PRIME_RL_PYTHON" eval_pi_checkpoints.py --base-url "http://localhost:$PORT/v1" --task "$TASK" \
  --lora-root "$RUN/adapters" --out "$OUT" --watch --exit-after-idle "${EXIT_AFTER_IDLE:-3600}" ${EXTRA_DRIVER_ARGS:-}
RC=$?
echo "[eval-server] driver exit $RC"
exit "$RC"
