#!/usr/bin/env bash
#SBATCH --job-name=lengthgen-frontier
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
# Example SLURM script; add --partition / --account for your cluster.
# From this experiment's directory: PRIME_API_KEY=... [PRIME_API_TEAM=...] sbatch slurm/frontier.sbatch   (hosted model, no GPUs)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the experiment directory}"
bash frontier/run_astra.sh
python frontier/collect_baseline.py
