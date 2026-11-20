#!/usr/bin/env bash
#SBATCH --job-name=synth-sft
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
# Example SLURM script; add --partition / --account for your cluster.
# From this experiment's directory: SFT_DATA_DIR=<dir> sbatch slurm/sft.sbatch   (data build: sft/README.md)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the experiment directory}"
REPO="$(cd ../.. && pwd)"
: "${SFT_DATA_DIR:?set SFT_DATA_DIR}"
bash "$REPO/training/sft/run_sft.sh" sft/configs/rlm-27b.toml
