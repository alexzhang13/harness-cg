#!/usr/bin/env bash
# Frontier reference (default GPT-6 Astra) on the six long eval splits, then aggregate.
#   PRIME_API_KEY=... [PRIME_API_TEAM=...] bash run_all.sh [task ...]   (MODEL=... for another model)
# run_astra.sh is the one-at-a-time version used for the paper.
set -euo pipefail
cd "$(dirname "$0")"
: "${PRIME_API_KEY:?set PRIME_API_KEY}"
TASKS=("$@")
[[ ${#TASKS[@]} -gt 0 ]] || TASKS=(oolong_pairs oolong graphwalks longbenchpro mrcr adaleval_bestanswer)
for t in "${TASKS[@]}"; do
  python run_frontier_eval.py --task "$t" --model "${MODEL:-openai/gpt-6-astra}" --concurrency "${CONC:-4}" \
    --timeout "${TIMEOUT:-0}" --request-timeout "${REQ_TIMEOUT:-1800}"
done
M="${MODEL:-openai/gpt-6-astra}"
python collect_baseline.py "results/${M##*/}"
