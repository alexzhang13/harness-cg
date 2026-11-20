#!/usr/bin/env bash
#SBATCH --job-name=gated-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
# Example SLURM script; add --partition / --account for your cluster.
# From this experiment's directory: MODEL=a1zhang/RLM-27B SERVED=rlm-27b TOKENIZER=Qwen/Qwen3.8-27B sbatch slurm/eval.sbatch <harness|hint|sft> [run_gated.py args]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the experiment directory}"
REPO="$(cd ../.. && pwd)"
TP="${TP:-2}" bash "$REPO/training/serving/serve_and_run.sh" bash eval.sh "$@"
