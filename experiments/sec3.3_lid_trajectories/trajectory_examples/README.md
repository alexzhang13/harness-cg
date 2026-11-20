# Appendix "Trajectory comparisons and examples"

`build_trajectory_examples.py` writes `trajectory_examples.tex` (the appendix's `appendix/trajectories/trajectory_examples.tex`)
and `trajectory_examples.json` (full root completions, hashes and all similarity scores). Each example pairs an
evaluation trajectory at the Fig. jaccard checkpoint with one of its 3 nearest training rollouts (Weighted Jaccard)
from `../neighbor_rewards.py`; the six pairs and their captions are the paper's hand-picked selection (archive step
and row per trajectory), so with new runs the `load(...)` arguments in `main` must be re-chosen.

```bash
pip install -r requirements.txt        # pinned black/Pygments: the Python cells are Black-formatted
python build_trajectory_examples.py --outputs $PRIME_RL_DIR/outputs --neighbors ../results/neighbor_rewards
```
