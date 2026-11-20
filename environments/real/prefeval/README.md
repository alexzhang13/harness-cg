# prefeval-real

PrefEval (classification task) as an `RLMTrainEnv` environment: a preference stated explicitly or through a
persona-driven dialogue is hidden among `inter_turns` filler turn-pairs; the final question offers four options.
Reward: exact match of the chosen letter. Preference position and filler turns are randomised per row.

Data: `amazon-science/PrefEval` @ `50795054`, raw files downloaded over HTTPS on first use into
`~/.cache/prefeval/5079505` (override with `PREFEVAL_DATA_DIR`, or point `PREFEVAL_REPO_DIR` at a clone).

Arguments used by the paper (`task="classification", pref_position="random", distractor_sampling="random"`, `seed=42`):
- train: `pref_form="implicit_persona", inter_turns=10, num_examples=-1`
- validation: `implicit_persona, inter_turns=150` and `explicit, inter_turns=300`, `num_examples=20`
