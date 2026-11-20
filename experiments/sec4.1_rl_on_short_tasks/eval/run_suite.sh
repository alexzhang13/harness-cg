#!/usr/bin/env bash
# Held-out evaluation suite (Tables eval-envs, main-composition): eight benchmarks, the served model as root and sub-LM.
#
#   MODEL=<served name> BASE_URL=http://127.0.0.1:<port>/v1 OUT=<dir> bash run_suite.sh
#
# BASE_URL is normally training/serving/lb_proxy.py over vLLM servers (262,144-token window). Optional:
#   BENCHMARKS="bcplus obliq corpusqa loogle elb longreason oolong ctc"   (default: all eight)
#   SMOKE=1               one item per benchmark (use a separate OUT)
#   LR_SUB_THINKING=1     LongReason sub-calls think with a 24,576-token budget (RLM-27B / RL rows; default);
#                         0 = thinking-off sub-calls (the Qwen3.8-27B base row)  -> OUT/longreason_{subthink,nothink}
#   OOLONG_THINKING=0     root + readers with {"enable_thinking": false} (paper rows; default);
#                         low = {"reasoning_effort": "low"}                       -> OUT/oolong_{nothink,low}
#   ACCT_TAG=<prefix>     tag requests per benchmark for lb_proxy.py token accounting (read by stats.py)
#   C_BCPLUS ... C_CTC    concurrent rollouts per benchmark
#   PYTHON                interpreter with verifiers + the env packages (default: python)
# ELB needs judge credentials (PRIME_API_KEY, or RLM_JUDGE_BASE_URL / RLM_JUDGE_API_KEY).
# Resumable. Scores: `python aggregate.py OUT`; statistics: `python stats.py OUT`.
set -uo pipefail
: "${MODEL:?set MODEL}" "${BASE_URL:?set BASE_URL}" "${OUT:?set OUT}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$(cd "$HERE/../../.." && pwd)"
PY="${PYTHON:-python}"
SMOKE=${SMOKE:-0}
BENCHMARKS=${BENCHMARKS:-"bcplus obliq corpusqa loogle elb longreason oolong ctc"}
LR_SUB_THINKING=${LR_SUB_THINKING:-1}
OOLONG_THINKING=${OOLONG_THINKING:-0}
ENVS="$H/environments"
NOTHINK='{"enable_thinking": false}'
SUB_NOTHINK='"sub_extra_body": {"chat_template_kwargs": {"enable_thinking": false}}'
# REPL timeouts for a shared server pool (sub-calls can queue behind other benchmarks); CTC raises them further.
export RLM_TRAIN_EXEC_TIMEOUT_S=${RLM_TRAIN_EXEC_TIMEOUT_S:-7200} RLM_TRAIN_SUBCALL_TIMEOUT_S=${RLM_TRAIN_SUBCALL_TIMEOUT_S:-3600} \
       RLM_TRAIN_REPL_REQUEST_TIMEOUT_S=${RLM_TRAIN_REPL_REQUEST_TIMEOUT_S:-14400}
export TOKENIZERS_PARALLELISM=false
export RLM_TRAIN_CODE_BLOCK_FEEDBACK=${RLM_TRAIN_CODE_BLOCK_FEEDBACK:-1}  # Section 4 setting: feedback on turns without a code block
# Sub-call concurrency per rollout: 24 BC+, 48 OBLIQ, 8 CTC, the harness default (16) elsewhere.
unset RLM_TRAIN_SUB_CONCURRENCY
OUT="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"
mkdir -p "$OUT/logs"
curl -sf -m5 "${BASE_URL%/v1}/health" >/dev/null || echo "WARNING: ${BASE_URL%/v1}/health is not 200"
echo "$(date -u +%FT%T) MODEL=$MODEL BASE_URL=$BASE_URL SMOKE=$SMOKE BENCHMARKS=$BENCHMARKS LR_SUB_THINKING=$LR_SUB_THINKING OOLONG_THINKING=$OOLONG_THINKING ACCT_TAG=${ACCT_TAG:-}" >> "$OUT/LAUNCH.log"

url() {   # url <name>: the endpoint for one benchmark (tagged when ACCT_TAG is set)
  if [[ -n ${ACCT_TAG:-} ]]; then echo "${BASE_URL%/v1}/t/$ACCT_TAG-$1/v1"; else echo "$BASE_URL"; fi
}
run() {   # run <log name> [VAR=value ...] <command or function ...>: background job with that environment, logged
  local name=$1 vars=(); shift
  while [[ $# -gt 0 && $1 == [A-Z_]*=* ]]; do vars+=("$1"); shift; done
  echo "[$name] ${vars[*]} $*" >> "$OUT/LAUNCH.log"
  ( for v in "${vars[@]}"; do export "$v"; done; "$@" ) >> "$OUT/logs/$name.log" 2>&1 < /dev/null &
}
run_eval() {   # run_eval <dir> <url> <env id> <args json> <run_eval options...>
  local d=$1 u=$2 id=$3 args=$4; shift 4
  mkdir -p "$d"
  "$PY" -u "$HERE/run_eval.py" --env-id "$id" --args-json "$args" --rollouts-per-example 1 \
    --base-url "$u" --model "$MODEL" --chat-template-kwargs "$NOTHINK" --out "$d/summary.jsonl" --cache-dir "$d/cache" "$@"
}

# ---- BrowseComp-Plus: all 830 queries, whole 100,195-document corpus (doc-store refs); orchestrator prompt, 40 turns, T=0.8,
#      root 16,384 tokens, sub-calls 65,536 tokens with thinking off, 4 h per rollout; substring grading.
bench_bcplus() {
  local ids=(--num-examples -1) c=${C_BCPLUS:-32}; [[ $SMOKE == 1 ]] && ids=(--num-examples -1 --ids 0) c=1
  run bcplus RUN_EVAL_SAVE_SUBCALLS=1 RLM_TRAIN_SUB_CONCURRENCY=24 PYTHONHASHSEED=0 HF_DATASETS_IN_MEMORY_MAX_SIZE=0 \
    PYTHONPATH="$H/src:$ENVS/real/browsecomp_plus" run_eval \
    "$OUT/bcplus" "$(url bcplus)" browsecomp-plus-real \
    "{\"split\": \"all\", \"docs_per_query\": \"all\", \"max_iterations\": 40, \"sub_max_tokens\": 65536, \"answer_judge\": false, \"orchestrator_prompt\": true, \"min_iterations\": 2, \"max_rollout_seconds\": 14400, $SUB_NOTHINK}" \
    "${ids[@]}" --max-completion-tokens 16384 --temperature 0.8 --request-timeout 1800 --max-concurrent "$c"
}

# ---- OBLIQ math: all 151 queries, whole corpus (pool 200,000), opaque ids, nDCG@50; orchestrator prompt, 20 turns, T=0.8,
#      root 16,384 tokens, sub-calls 65,536 tokens with thinking off, 4 h per rollout.
bench_obliq() {
  local ids=(--num-examples 151) c=${C_OBLIQ:-32}; [[ $SMOKE == 1 ]] && ids=(--num-examples 151 --ids 0) c=1
  run obliq RUN_EVAL_SAVE_SUBCALLS=1 RLM_TRAIN_SUB_CONCURRENCY=48 PYTHONHASHSEED=0 HF_DATASETS_IN_MEMORY_MAX_SIZE=0 \
    PYTHONPATH="$H/src:$ENVS/real/obliq" run_eval \
    "$OUT/obliq" "$(url obliq)" obliq-real \
    "{\"task\": \"math\", \"opaque_ids\": true, \"seed\": 74, \"pool_size\": 200000, \"min_distractors\": 10, \"metric\": \"ndcg@50\", \"doc_max_chars\": 4000, \"max_iterations\": 20, \"sub_max_tokens\": 65536, \"orchestrator_prompt\": true, \"min_iterations\": 2, \"max_rollout_seconds\": 14400, $SUB_NOTHINK}" \
    "${ids[@]}" --max-completion-tokens 16384 --temperature 0.8 --request-timeout 1800 --max-concurrent "$c"
}

# ---- CorpusQA 1M real_estate_en, 81 questions: orchestrator prompt, 30 turns, T=0, root 8,192 tokens, thinking off root + subs.
bench_corpusqa() {
  local ids=(--num-examples -1) c=${C_CQA:-16}; [[ $SMOKE == 1 ]] && ids=(--num-examples -1 --ids 0) c=1
  run corpusqa RUN_EVAL_SAVE_SUBCALLS=1 PYTHONPATH="$H/src:$ENVS/real/corpusqa" run_eval \
    "$OUT/corpusqa" "$(url corpusqa)" corpusqa-real \
    "{\"set_name\": \"1m\", \"domain\": \"real_estate_en\", \"max_iterations\": 30, \"orchestrator_prompt\": true, \"min_iterations\": 2, $SUB_NOTHINK}" \
    "${ids[@]}" --max-completion-tokens 8192 --temperature 0 --request-timeout 3600 --max-concurrent "$c"
}

# ---- LooGLE-v2, the fixed 206-item cohort (>= 500K tokens): 12 turns, T=0.2, 6,000 tokens per turn, <= 80 sub-calls,
#      2 h per item. Driver: environments/synthetic/loogle_v2/tools/strategy_lab.py --arm root.
bench_loogle() {
  local extra=(--concurrency "${C_LOOGLE:-12}"); [[ $SMOKE == 1 ]] && extra=(--concurrency 1 --indices 554)
  run loogle PYTHONPATH="$H/src:$ENVS/real/loogle_v2:$ENVS/synthetic/synth_common:$ENVS/synthetic/loogle_v2" \
    "$PY" -u "$ENVS/synthetic/loogle_v2/tools/strategy_lab.py" --arm root --base-url "$(url loogle)" --model "$MODEL" \
    --max-turns 12 --max-tokens 6000 --temperature 0.2 --timeout 7200 --request-timeout 7200 "${extra[@]}" --out "$OUT/loogle"
}

# ---- ExpertLongBench T01 0-99 / T03 0-49 / T04 0-99 / T06 0-99 in 25-id shards: orchestrator prompt, 40 turns, T=0, root 8,192,
#      thinking off; GPT-4.1-mini checklist judge. Shards with output are skipped; elb/redo.jobs lines are run too.
bench_elb() {
  local d="$OUT/elb" jobs=() c=${C_ELB:-8}; mkdir -p "$d"
  if [[ $SMOKE == 1 ]]; then jobs=("T03-s T03MaterialSEG 0"); c=1
  else
    for spec in "T01LegalMDS 0-24 25-49 50-74 75-99" "T03MaterialSEG 0-24 25-49" "T04EduPAE 0-24 25-49 50-74 75-99" \
                "T06HealthCNG 0-24 25-49 50-74 75-99"; do
      set -- $spec; local task=$1; shift
      for r in "$@"; do jobs+=("${task:0:3}-$r $task $r"); done
    done
    [[ -f $d/redo.jobs ]] && while read -r line; do [[ -n $line ]] && jobs+=("$line"); done < "$d/redo.jobs"
  fi
  for j in "${jobs[@]}"; do
    set -- $j
    [[ -s $d/$1.jsonl ]] && continue
    run "elb-$1" PYTHONPATH="$H/src:$ENVS/real/expertlongbench" elb_shard "$1" "$2" "$3" "$c"
  done
}
elb_shard() {   # elb_shard <name> <task> <ids> <concurrency>; judge credentials: PRIME_API_KEY or RLM_JUDGE_*
  "$PY" -u "$HERE/elb.py" --task "$2" --ids "$3" --base-url "$(url "elb-${2:0:3}")" --model "$MODEL" \
    --concurrency "$4" --tag "$1" --out-dir "$OUT/elb"
}

# ---- LongReason 128k, 794 questions: 40 turns, T=0, root 4,096 tokens, root thinking off; sub-calls per LR_SUB_THINKING.
bench_longreason() {
  local ids=(--num-examples -1) c=${C_LR:-32} d args; [[ $SMOKE == 1 ]] && ids=(--num-examples -1 --ids 0) c=1
  if [[ $LR_SUB_THINKING == 1 ]]; then d="$OUT/longreason_subthink"
    args='{"split": "128k", "max_iterations": 40, "sub_max_tokens": 24576, "sub_extra_body": {"chat_template_kwargs": {"enable_thinking": true}}}'
  else d="$OUT/longreason_nothink"; args="{\"split\": \"128k\", \"max_iterations\": 40, $SUB_NOTHINK}"; fi
  run longreason RUN_EVAL_SAVE_SUBCALLS=1 PYTHONPATH="$H/src:$ENVS/real/longreason" run_eval \
    "$d" "$(url longreason)" longreason-real "$args" \
    "${ids[@]}" --max-completion-tokens 4096 --temperature 0 --request-timeout 3600 --max-concurrent "$c"
}

# ---- OOLONG-Real D&D test, 577 questions over ten 3-episode windows: 40 turns, T=0, root and reader 4,096 tokens,
#      <= 600 sub-calls, 2 h per question; sub-call settings must not be overridden (RLM_TRAIN_SUB_* unset).
bench_oolong() {
  local d ctk extra=(--concurrency "${C_OOLONG:-48}")
  case $OOLONG_THINKING in
    0) d="$OUT/oolong_nothink" ctk=$NOTHINK ;;
    low) d="$OUT/oolong_low" ctk='{"reasoning_effort": "low"}' ;;
    *) echo "OOLONG_THINKING must be 0 or low" >&2; return 1 ;;
  esac
  [[ $SMOKE == 1 ]] && extra=(--concurrency 1 --limit 1)
  run oolong PYTHONPATH="$H/src:$ENVS/synthetic/synth_common:$ENVS/synthetic/oolong_real" \
    oolong_run --out "$d" --base-url "$(url oolong)" --model "$MODEL" \
    --root-max-tokens 4096 --reader-max-tokens 4096 --chat-template-kwargs "$ctk" --timeout 7200 \
    --tokenizer "${OOLONG_TOKENIZER:-Qwen/Qwen3.8-27B}" "${extra[@]}"
}
oolong_run() {
  for v in ${!RLM_TRAIN_SUB_@}; do unset "$v"; done
  "$PY" -u "$HERE/oolong_dnd.py" "$@"
}

# ---- CTC-Bench, five cells (CTC-Core v1 test rows + the 60 frozen grouping rows): 30 turns, root and sub-calls
#      uncapped with thinking off; core cells T=0.7 without the orchestrator prompt, grouping T=0 with it; long REPL timeouts.
bench_ctc() {
  local c=${C_CTC:-24}
  local sub="\"sub_max_tokens\": null, $SUB_NOTHINK"
  for cell in qdmatch_hpqa@r256k contradiction_iid@r1m outlier@r1m xabsence_paraphrase@r32k grouping@r32k; do
    local id=ctc-core-env T=0.7 args="{\"cells\": \"$cell\", \"part\": \"test\", \"max_iterations\": 30, $sub}" ids=()
    if [[ $cell == grouping@r32k ]]; then
      id=ctcbench-real T=0
      args="{\"task\": \"grouping\", \"split\": \"eval\", \"rungs\": \"r32k\", \"ids\": \"grouping32_test\", \"max_iterations\": 30, \"orchestrator_prompt\": true, $sub}"
    fi
    if [[ $SMOKE == 1 ]]; then
      [[ $cell == xabsence_paraphrase@r32k || $cell == grouping@r32k ]] || continue
      local first=0
      [[ $cell == grouping@r32k ]] && first=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['test'][0])" "$ENVS/real/CTCBench/ctcbench_real/grouping32_v1.json")
      ids=(--ids "$first"); c=1
    fi
    run "ctc-$cell" RUN_EVAL_SAVE_SUBCALLS=1 RLM_TRAIN_SUB_CONCURRENCY=8 \
      RLM_TRAIN_SUBCALL_TIMEOUT_S=10800 RLM_TRAIN_EXEC_TIMEOUT_S=14400 RLM_TRAIN_REPL_REQUEST_TIMEOUT_S=30000 \
      PYTHONPATH="$H/src:$ENVS/real/CTCBench" run_eval \
      "$OUT/ctc/$cell" "$(url "ctc-${cell/@/_}")" "$id" "$args" \
      --num-examples -1 "${ids[@]}" --max-completion-tokens 0 --temperature "$T" --request-timeout 10800 --max-concurrent "$c"
  done
}

for b in $BENCHMARKS; do
  declare -f "bench_$b" >/dev/null || { echo "unknown benchmark: $b" >&2; exit 1; }
  "bench_$b"
done
echo "launched: $BENCHMARKS -> $OUT (logs: $OUT/logs); waiting"
wait
echo "$(date -u +%FT%T) all benchmark processes exited" >> "$OUT/LAUNCH.log"
PYTHONPATH="$H/src:$ENVS/real/browsecomp_plus" "$PY" "$HERE/aggregate.py" "$OUT" || true
