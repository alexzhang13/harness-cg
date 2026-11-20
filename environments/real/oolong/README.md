# oolong-real

OOLONG-synth aggregation QA as an `RLMTrainEnv` environment. The context is a window of unlabeled
`Date || User || Instance` lines; the question asks for an aggregate over the hidden labels. Reward: exact match
on the final answer (after the last `Answer:` / `Label:` marker), per answer type; see `_synth_score`.

Data: HF `oolongbench/oolong-synth`, `validation` split (task families `trec_coarse` and `spam`), downloaded on
first use at the revision the paper's runs used (`f0d59eaf`; `OOLONG_SYNTH_REVISION` overrides; `datasets` ignores the
revision when `HF_HUB_OFFLINE=1`, so download it online once).

Arguments used by the paper (`seed=42`, `exclude_numeric=false`, `max_iterations=20`):

| use | `dataset_name` | `min_ctx` / `max_ctx` | `num_examples` |
|---|---|---|---|
| length generalization: train / eval | `trec_coarse` | 32768 / 262144 (single lengths) | -1 (50 rows) / 50 |
| strategy generalization: eval | `spam` | 131072 | 50 |
| broad mixture: train (both families) / eval | both | 8192-65536 / `trec_coarse` 262144 and 100000-131072, `spam` 131072 | -1 / 20 |
