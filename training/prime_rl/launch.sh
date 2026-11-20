#!/usr/bin/env bash
# Run a prime-rl RL config (trainer + vLLM inference + orchestrator on one node).
#
#   PRIME_RL_DIR=~/prime-rl bash training/prime_rl/launch.sh <config.toml> [extra `rl` args]
#
# Reads credentials from $PRIME_RL_DIR/.env and resumes from existing checkpoints. OUTPUT_DIR overrides the
# config's output_dir; PRIME_RL_PYTHON bypasses `uv run`. ${REPO}, ${ADALEVAL_DATA_DIR}, ${OOLONG_REAL_DATA_DIR},
# ${INIT_WEIGHTS} and ${SFT_DATA_DIR} in configs are substituted.
set -euo pipefail

SRC_CONFIG="$(realpath "$1")"; shift
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
[[ -f "$SRC_CONFIG" ]] || { echo "config not found: $SRC_CONFIG" >&2; exit 1; }
[[ -d "$PRIME_RL_DIR" ]] || { echo "PRIME_RL_DIR not found: $PRIME_RL_DIR" >&2; exit 1; }

export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ADALEVAL_DATA_DIR="${ADALEVAL_DATA_DIR:-$HOME/ada_leval_data}"
CONFIG="$(mktemp --suffix=.toml)"
VARS='${REPO} ${ADALEVAL_DATA_DIR} ${OOLONG_REAL_DATA_DIR} ${INIT_WEIGHTS} ${SFT_DATA_DIR}'
for v in $(grep -oE '\$\{[A-Z_]+\}' "$SRC_CONFIG" | sort -u | tr -d '${}'); do
  [[ -n "${!v:-}" ]] || { echo "config needs \$$v; export it first" >&2; exit 1; }
done
envsubst "$VARS" < "$SRC_CONFIG" > "$CONFIG"

if [[ -f "$PRIME_RL_DIR/.env" ]]; then
  set -a; source "$PRIME_RL_DIR/.env"; set +a
fi

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-900}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_$USER}"
mkdir -p "$TRITON_CACHE_DIR"

RESUME=()
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  RESUME=(--output-dir "$OUTPUT_DIR")
fi
if grep -qE '^\[trainer\.ckpt\]' "$CONFIG"; then
  OUT_REL="${OUTPUT_DIR:-$(grep -aoE '^output_dir = "[^"]+"' "$CONFIG" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')}"
  if compgen -G "$PRIME_RL_DIR/$OUT_REL/*/checkpoints/step_*" >/dev/null 2>&1 \
     || compgen -G "$PRIME_RL_DIR/$OUT_REL/checkpoints/step_*" >/dev/null 2>&1; then
    RESUME+=(--ckpt.resume-step -1)
  fi
fi

cd "$PRIME_RL_DIR"
if [[ -n "${PRIME_RL_PYTHON:-}" ]]; then
  exec "$PRIME_RL_PYTHON" "$(dirname "$PRIME_RL_PYTHON")/rl" @ "$CONFIG" "${RESUME[@]}" "$@"
fi
exec uv run rl @ "$CONFIG" "${RESUME[@]}" "$@"
