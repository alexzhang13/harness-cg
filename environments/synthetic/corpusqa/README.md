# corpusqa-synth

Synthetic CorpusQA: generated corpora of periodic monthly statistical releases (adjusted / not-adjusted table pairs,
revised prints, newest release wins) with a CorpusQA-style aggregation question and an exact gold; an entity-report
family (10-Q-like filings, school data sets) also exists. No external data. `sub_lm="real"` sends sub-calls to the
root's own model (thinking off); `sub_lm="oracle"` uses the programmed `CorpusOracle`.

- RL (`rlstage-corpusqa`): `family="periodic", surface="varied", scale="1m"` and `scale="128k"` (one stream each),
  `num_examples=2000, sub_lm="real", hint="none", hint_prob=0.0, reward_mode="outcome"`.
- SFT trace generation: `split="train", family="periodic", surface="varied", sub_lm="real", max_iterations=20`,
  scales 128k / 1m, `hint="program", hint_prob=1.0` (package hint `corpusqa_synth/hint_periodic.md`) or
  `hint="strategy", hints=HintSpec(strategy=<note file>)`.
- Splits: `train` / `val` / `test` seed ranges (disjoint).
