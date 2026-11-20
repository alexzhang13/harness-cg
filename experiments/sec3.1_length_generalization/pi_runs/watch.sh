#!/usr/bin/env bash
# Every 10 minutes, submit missing Pi training runs and eval servers (resubmits preempted runs).
#
#   nohup bash watch.sh > watch.log 2>&1 &
#
# TASKS, SEEDS, ORDER, MAX_QUEUED (3), EVAL_JOBS (4), EVAL_SBATCH_ARGS (extra sbatch args for eval servers)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
TASKS="${TASKS:-longbenchpro,mrcr}"; SEEDS="${SEEDS:-0}"; ORDER="${ORDER:-task}"; MAX_QUEUED="${MAX_QUEUED:-3}"; EVAL_JOBS="${EVAL_JOBS:-4}"
EVAL_SBATCH_ARGS="${EVAL_SBATCH_ARGS-}"
while true; do
  echo "[$(date '+%F %T')]"
  "$PY" "$HERE/submit.py" --tasks "$TASKS" --seeds "$SEEDS" --order "$ORDER" --max-queued "$MAX_QUEUED"
  "$PY" "$HERE/submit_eval.py" --tasks "$TASKS" --seeds "$SEEDS" --servers-per-run 2 --max-jobs "$EVAL_JOBS" --sbatch-args="$EVAL_SBATCH_ARGS"
  sleep 600
done
