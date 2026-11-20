#!/usr/bin/env bash
# Train one Pi run on the node this runs on (8 GPUs: 6 inference + 2 trainer).
#
#   bash run.sh <task> <seed> [extra `rl` args]
#     task: mrcr | longbenchpro | oolong-pairs | adaleval      seed: 0
#
# Skips a complete run, else resumes from the newest complete checkpoint. Adapters are copied to <run>/adapters/
# while training. Exit status 0 means the run is complete.
set -uo pipefail
TASK="${1:?task}"; SEED="${2:?seed}"; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh" || exit 1

RUN="$("$PRIME_RL_PYTHON" "$HERE/settings.py" run-dir "$TASK" "$SEED")"
ck() { "$PRIME_RL_PYTHON" "$HERE/ckpt_tools.py" "$@"; }

if ck complete "$RUN"; then echo "[run] $TASK seed $SEED is complete -> skip"; exit 0; fi
[[ -f "$HERE/configs/$TASK/seed$SEED.toml" ]] || { echo "[run] no config for $TASK seed $SEED" >&2; exit 1; }

mkdir -p "$RUN/logs"
cp "$HERE/configs/$TASK/seed$SEED.toml" "$RUN/config.toml"
"$PRIME_RL_PYTHON" "$HERE/settings.py" snapshot "$TASK" "$SEED" > "$RUN/settings.json"
git -C "$HCG" rev-parse HEAD > "$RUN/code_rev.txt" 2>/dev/null || true
# hashes of the source files, including uncommitted changes
( cd "$HCG" && { git status --short 2>/dev/null; find src/rlm_train environments experiments/sec3.1_length_generalization/pi_runs \
    -type f \( -name '*.py' -o -name '*.ts' -o -name '*.patch' -o -name '*.sh' -o -name '*.txt' \) \
    -not -path '*/__pycache__/*' -not -path '*/node_modules/*' | sort | xargs sha256sum; } > "$RUN/code_files.txt" ) 2>/dev/null || true
rm -f "$RUN/.sync_stop"

STEP="$(ck prepare-resume "$RUN")"
RESUME=()
if [[ -n "$STEP" ]]; then RESUME=(--ckpt.resume-step -1); echo "[run] resuming from step $STEP"; fi

# torch and Python hash seeds (the config sets the orchestrator and vLLM seeds)
export PRIME_RL_TORCH_SEED="$SEED" PYTHONHASHSEED="$SEED"
export PYTHONPATH="$HERE/seed:$PYTHONPATH"

echo "[run] $TASK seed $SEED -> $RUN on $(hostname) gpus=${CUDA_VISIBLE_DEVICES:-all}"
ck sync-loop "$RUN" & SYNC_PID=$!
cd "$PRIME_RL_DIR"
timeout "${RUN_TIMEOUT:-13h}" "$RL" @ "$RUN/config.toml" --output-dir "$RUN" "${RESUME[@]}" "$@" 2>&1 | tee -a "$RUN/logs/train.log"
RC=${PIPESTATUS[0]}
kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null

ck finalize "$RUN"; FIN=$?
echo "[run] rl exit=$RC complete=$([[ $FIN -eq 0 ]] && echo yes || echo no)"
exit "$FIN"
