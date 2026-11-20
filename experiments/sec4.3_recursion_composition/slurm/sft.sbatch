#!/usr/bin/env bash
#SBATCH --job-name=gated-sft
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=600G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
# Example SLURM script; add --partition / --account for your cluster.
# From this experiment's directory: PRIME_RL_SFT_GATED_DIR=<checkout> SFT_DATA_DIR=<dir> [INIT_WEIGHTS=<RLM-27B dir>] sbatch slurm/sft.sbatch <rlm-27b|qwen38-27b>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the experiment directory}"
bash sft/run_sft.sh "$@"
