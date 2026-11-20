#!/usr/bin/env bash
#SBATCH --job-name=lengthgen-eval-lmrepl
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
# Example SLURM script; add --partition / --account for your cluster.
# From this experiment's directory: sbatch slurm/eval_lm_repl.sbatch oolong
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the experiment directory}"
bash eval_lm_repl.sh "$@"
