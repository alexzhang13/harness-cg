#!/bin/bash
# Build CTC train rows with the upstream generators (pip install "git+https://github.com/PrasannS/corpustaskcomplexity#subdirectory=ctc").
# Overlap with the eval ladders is removed by the env loader.
#   usage: build_train.sh [task ...]      (N rows per task, default 300) -> $CTCBENCH_TRAIN_DIR/<task>/train.jsonl
CTC=${CTC_DATA_BIN:-ctc-data}
OUT=${OUT:-${CTCBENCH_TRAIN_DIR:-$HOME/ctcbench_data/train}}
LOG=${LOG:-$OUT/../logs}; mkdir -p "$LOG"
N=${N:-300}
TASKS=${@:-textgroups strmatch contradiction nq hotpotqa fiqa scifact rerank outlier_review absence reorder xabsence grouping_labeled qdmatch_nq qdmatch_hpqa outlier}
for t in $TASKS; do
  case $t in
    textgroups|strmatch) POOL=() ;;
    *) POOL=(--pool auto) ;;
  esac
  case $t in
    absence|reorder) RUNGS=8k,16k ;;
    *) RUNGS=8k,16k,32k ;;
  esac
  echo "[$(date +%T)] $t start" ; 
  $CTC build --task $t --split train --train $N --rungs $RUNGS "${POOL[@]}" --out $OUT --force > $LOG/$t.log 2>&1
  echo "[$(date +%T)] $t exit=$? $(tail -1 $LOG/$t.log)"
done
