#!/usr/bin/env bash
# Hinted trick_synth rollouts (notes map_flag_v7a / v7b). Needs a served teacher reachable through
# OPENAI_BASE_URL (OpenAI-compatible; served model name in MODEL). Run from the repository root:
#   OPENAI_BASE_URL=http://localhost:8000/v1 OUT=outputs/synth-datagen/obliq bash synth-datagen/obliq/run_gen.sh
set -euo pipefail
OUT=${OUT:-outputs/synth-datagen/obliq}; MODEL=${MODEL:-qwen38-27b}
# Sampling: probe defaults (T=0.7, 6000 tokens) with --no-think.
SAMPLING=${SAMPLING:---no-think}
export PYTHONPATH=src:environments/synthetic/synth_common:environments/synthetic/trick${PYTHONPATH:+:$PYTHONPATH}
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
export RLM_TRAIN_CODE_BLOCK_FEEDBACK=${RLM_TRAIN_CODE_BLOCK_FEEDBACK:-1}  # Section 4 setting: feedback on turns without a code block
args() {  # $1 = note file, $2 = read JSON
  python3 -c 'import json,sys; print(json.dumps({"split": "train", "num_examples": 10000, "read": json.loads(sys.argv[2]),
    "user_prologue": "<hint>\n" + open(sys.argv[1]).read() + "\n</hint>"}))' "$1" "$2"
}
run() {  # $1 = note, $2 = read JSON, $3 = out dir, $4 = first seed, $5 = n
  python synth-datagen/obliq/gen_traces.py --env trick_synth:load_environment --env-args "$(args "synth-datagen/obliq/notes/$1.md" "$2")" \
    --model "$MODEL" --start "$4" --n "$5" --concurrency 16 $SAMPLING --out "$OUT/$3"
}
FAULTS='{"preset": "real_v1", "fault_q": 0.15, "fault_desc": 0.15, "fault_flag": 0.15}'
run map_flag_v7a "$FAULTS" trick_gen9/v7a_6400 6400 75;   run map_flag_v7b "$FAULTS" trick_gen9/v7b_6475 6475 75
run map_flag_v7a "$FAULTS" trick_gen9/v7a_6550 6550 40;   run map_flag_v7b "$FAULTS" trick_gen9/v7b_6590 6590 40
run map_flag_v7b "$FAULTS" trick_gen10/v7b_6630 6630 90;  run map_flag_v7b "$FAULTS" trick_gen10/v7b_6720 6720 90
run map_flag_v7a "$FAULTS" trick_gen10/v7a_6810 6810 60
run map_flag_v7a "$FAULTS" trick_probe/pilot_v7a_9000 9000 12
run map_flag_v7b "$FAULTS" trick_probe/pilot_v7b_9100 9100 12;  run map_flag_v7b "$FAULTS" trick_probe/pilot_v7b_9200 9200 12
