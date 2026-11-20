#!/usr/bin/env bash
# prime-rl SFT on per-turn rows (build_turn_mix.py), with no-split packing and loss on the last turn only.
#
#   SFT_DATA_DIR=<dir with the built dataset> bash training/sft/run_sft.sh <config.toml> [extra `sft` args]
#
# Resumes from the latest checkpoint. TRAIN_GPUS selects GPUs.
set -euo pipefail
SRC_CONFIG="$(realpath "$1")"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO="$(cd "$HERE/../.." && pwd)"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
if [[ -f "$PRIME_RL_DIR/.env" ]]; then set -a; source "$PRIME_RL_DIR/.env"; set +a; fi

CONFIG="$(mktemp --suffix=.toml)"
for v in $(grep -oE '\$\{[A-Z_]+\}' "$SRC_CONFIG" | sort -u | tr -d '${}'); do
  [[ -n "${!v:-}" ]] || { echo "config needs \$$v; export it first" >&2; exit 1; }
done
envsubst '${REPO} ${SFT_DATA_DIR} ${INIT_WEIGHTS}' < "$SRC_CONFIG" > "$CONFIG"

OUT_REL="$(grep -aoE '^output_dir = "[^"]+"' "$CONFIG" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
RESUME=()
if compgen -G "$PRIME_RL_DIR/$OUT_REL/checkpoints/step_*" >/dev/null 2>&1; then RESUME=(--ckpt.resume-step -1); fi

export RLM_SFT_PATCH=1 RLM_SFT_LAST_TURN_ONLY=1
export PYTHONPATH="$HERE/patches${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false HF_DATASETS_IN_MEMORY_MAX_SIZE=0
[[ -n "${TRAIN_GPUS:-}" ]] && export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"

cd "$PRIME_RL_DIR"
if [[ -n "${PRIME_RL_PYTHON:-}" ]]; then
  exec "$(dirname "$PRIME_RL_PYTHON")/sft" @ "$CONFIG" "${RESUME[@]}" "$@"
fi
exec uv run sft @ "$CONFIG" "${RESUME[@]}" "$@"
