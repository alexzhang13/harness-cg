# golongrl-real

GoLongRL (Kwai-Klear) as an `RLMTrainEnv` environment, restricted to the two families the paper trains on:
`table` (multi-table extraction, row-multiset IoU) and `summary` (LongBench-Pro summaries, ROUGE-L). Rows whose
own gold answer does not score >= 0.999 are dropped. Reward ports are in `golongrl_real/metrics.py`.

Data: HF `Kwai-Klear/GoLongRL` (11 parquet shards of the `train` split, downloaded on first use).

Arguments used by the paper (`language="English"`, `num_examples=-1`, `seed=42`):
- `task="table", min_context_chars=20000`
- `task="summary"` (default `min_context_chars=200`)
