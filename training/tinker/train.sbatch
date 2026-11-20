#!/usr/bin/env bash
# SLURM wrapper for a Tinker run (CPU only; no GPUs). Runs any section launcher with BACKEND=tinker:
#
#   sbatch -p <partition> training/tinker/train.sbatch experiments/sec3.1_length_generalization/run_train.sh rlm oolong
#   sbatch training/tinker/train.sbatch experiments/appendix/broad_mixture/run_train.sh --max-steps 2
#
# Submit from the repository root; a requeued job resumes. Variables as for training/tinker/launch.sh.
#SBATCH --job-name=tinker-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --time=7-00:00:00
#SBATCH --output=tinker-train-%j.out
set -euo pipefail
export BACKEND=tinker
exec bash "$@"
