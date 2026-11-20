#!/usr/bin/env bash
#SBATCH --job-name=similarity
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
# Example SLURM script; add --partition / --account for your cluster.
# From this experiment's directory: sbatch slurm/similarity.sbatch   (CPU; reads the section 3 run directories)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the experiment directory}"
python run_distance.py --all
python run_distance.py --panel oolong-pairs --harnesses pi --cand-k 32 --weighted-jaccard exact
python run_distance.py --panel obliq_twitter_to_wildchat --harnesses rlm --max-eval-step 150
python neighbor_rewards.py --workers 8
