#!/usr/bin/env bash
# Train an experiment config on Tinker: the same TOML that training/prime_rl/launch.sh runs.
#
#   TINKER_API_KEY=... bash training/tinker/launch.sh <config.toml> [rlm-train-tinker train args]
#
# CPU only. Runs go to $TINKER_RUNS_DIR/<output_dir>-seed<SEED> (default ~/tinker-runs; LOG_DIR overrides) and
# resume from their last checkpoint. Also: TINKER_PYTHON, TINKER_ENV_FILE, TINKER_BASE_MODEL.
set -euo pipefail

SRC_CONFIG="$(realpath "$1")"; shift
[[ -f "$SRC_CONFIG" ]] || { echo "config not found: $SRC_CONFIG" >&2; exit 1; }
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ADALEVAL_DATA_DIR="${ADALEVAL_DATA_DIR:-$HOME/ada_leval_data}"
if [[ -n "${TINKER_ENV_FILE:-}" ]]; then
  set -a; source "$TINKER_ENV_FILE"; set +a
fi
[[ -n "${TINKER_API_KEY:-}" ]] || { echo "TINKER_API_KEY is not set" >&2; exit 1; }

PY="${TINKER_PYTHON:-python}"
SEED="${SEED:-42}"
if [[ -z "${LOG_DIR:-}" ]]; then
  OUT="$(grep -aoE '^output_dir = "[^"]+"' "$SRC_CONFIG" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
  LOG_DIR="${TINKER_RUNS_DIR:-$HOME/tinker-runs}/${OUT:-$(basename "$SRC_CONFIG" .toml)}-seed$SEED"
fi
EXTRA=()
[[ -n "${TINKER_BASE_MODEL:-}" ]] && EXTRA+=(--base-model "$TINKER_BASE_MODEL")
echo "[tinker] $SRC_CONFIG -> $LOG_DIR"
exec "$PY" -m rlm_train_tinker.cli train "$SRC_CONFIG" --log-dir "$LOG_DIR" --seed "$SEED" "${EXTRA[@]}" "$@"
