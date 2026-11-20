#!/bin/bash
# Second train build (seed 43) into <train dir>/../train_extra/; merge_extra.py appends it to the first.
CTC=${CTC_DATA_BIN:-ctc-data}
TRAIN=${CTCBENCH_TRAIN_DIR:-$HOME/ctcbench_data/train}
OUT=${OUT:-$TRAIN/../train_extra}; LOG=${LOG:-$TRAIN/../logs}; mkdir -p "$LOG"
for spec in "nq 1000" "hotpotqa 1000" "strmatch 1000" "rerank 1000" "xabsence 1000" "absence 1000" "textgroups 1000" "fiqa 3000" "qdmatch_nq 1000"; do
  set -- $spec; t=$1; n=$2
  case $t in textgroups|strmatch) POOL=() ;; *) POOL=(--pool auto) ;; esac
  case $t in absence|reorder) RUNGS=8k,16k ;; *) RUNGS=8k,16k,32k ;; esac
  echo "[$(date +%T)] $t start n=$n"
  $CTC build --task $t --split train --train $n --seed 43 --rungs $RUNGS "${POOL[@]}" --out $OUT --force > $LOG/extra_$t.log 2>&1
  echo "[$(date +%T)] $t exit=$? $(tail -1 $LOG/extra_$t.log)"
done
