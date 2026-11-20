#!/usr/bin/env bash
# Frontier reference line of Fig. length-gen: GPT-6 Astra on the six long eval splits, one datapoint at a time.
# Then `python collect_baseline.py`.
#   PRIME_API_KEY=... [PRIME_API_TEAM=<team id>] bash run_astra.sh
# Creates a venv at $PYTHON (default <repo>/.venv).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

: "${PRIME_API_KEY:?Set PRIME_API_KEY (Prime Inference key)}"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  command -v uv >/dev/null || { echo "run_astra.sh: uv is required to create $PYTHON" >&2; exit 2; }
  uv venv "$ROOT/.venv"
fi
uv pip install -q --python "$PYTHON" 'verifiers==0.1.14' -e "$ROOT" \
  -e "$ROOT/environments/real/mrcr_v2" \
  -e "$ROOT/environments/real/graphwalks" \
  -e "$ROOT/environments/real/longbenchpro" \
  -e "$ROOT/environments/real/oolong" \
  -e "$ROOT/environments/real/oolong_pairs" \
  -e "$ROOT/environments/real/ada_leval"

MODEL="openai/gpt-6-astra"
RESULTS_DIR="$HERE/results/gpt-6-astra"
LOG_DIR="$HERE/logs/gpt-6-astra"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

exec "$PYTHON" run_frontier_eval.py --task all --model "$MODEL" --concurrency 1 \
  --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG_DIR/all.log"
