# corpusqa-real

CorpusQA (`Tongyi-Zhiwen/CorpusQA`, auto-downloaded from the HF Hub; the `1m` file is about 1 GB) as an
`RLMTrainEnv`: cross-document aggregation questions over a corpus of 4-9 (`set_name="128k"`) or 25-90
(`set_name="1m"`) documents, 329 questions per set over four domains. `context` is a list with one string per
document; the upstream output requirements are shown in the root prompt.

- Reward by gold type: list -> set-F1 over entity names, number -> match within `num_tolerance`, string ->
  normalised exact match.
- Rows are filtered by `domain`, shuffled with `seed` (default 42) and cut to `num_examples`; ids are positions in
  that order.
- Held-out evaluation: `set_name="1m", domain="real_estate_en"` (81 questions, ids 0-80, default seed), with
  `max_iterations=30, orchestrator_prompt=true`.
- RL validation streams: `set_name="128k", domain="financial_en", num_examples=13, seed=7` and
  `set_name="128k", domain="education_en", num_examples=12, seed=7`.
