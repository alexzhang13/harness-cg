# loogle-strategy-synth (+ the LooGLE-v2 eval driver)

Synthetic LooGLE-v2: procedural fictional 10-K filings, Python repositories with call graphs, game logs and version
trees, with LooGLE-v2 question types, answer formats and scoring. No external data. Sub-calls go to the root's own
model (thinking off). Reward: outcome, or (default `reward_mode="gated"`) outcome only within the efficiency budgets.

- RL (`rlstage-loogle`): `split="train", family="mix", num_examples=2000, note="", reward_mode="outcome",
  generator_version=2, max_iterations=30`.
- SFT trace generation: one family per run (`finance` / `code` / `game` / `versions`), `split="train"`,
  `reward_mode="outcome"`, a program note as `note=` (prepended inside `<strategy_hint>` tags, stripped for SFT).
- Splits: `train` / `val` / `test` seed ranges.

`tools/strategy_lab.py` is the LooGLE-v2 evaluation driver (real rows via `loogle_v2_real`, cohort
`tools/eval_cohort.json` = the 206 test rows with >= 2M-character documents):
`--arm root --max-turns 12 --max-tokens 6000 --temperature 0.2 --timeout 7200 --request-timeout 7200`.
