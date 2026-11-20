# obliq-bench-real (env id `obliq-real`)

OBLIQ-Bench retrieval as an `RLMTrainEnv` environment. Each query gets a candidate pool (its gold positives plus
random distractors) as the `context` list; the model ranks the doc ids. Reward: nDCG@50 against the qrels
restricted to the pool; nDCG@10/50 and recall@10/50/100 are logged as `obliq_*` metrics (the paper plots
`obliq_ndcg10`). Configs must use `id = "obliq-real"` (verifiers resolves the module `obliq_real` from the id).

Data: HF `dianetc/OBLIQ-Bench` v1.0 (pinned revision `4ebee29f`, which the paper's runs used; `OBLIQ_BENCH_REVISION` overrides), fetched on first use (may need `HF_TOKEN`); the `wildchat` / `congress` corpora
are streamed and read up to `max_scan` rows (default 250000).

Arguments used by the paper (`pool_size=300, min_distractors=10, metric="ndcg@50", doc_max_chars=4000, seed=42,
max_iterations=20`, plus `min_iterations=1, max_rollout_seconds=1800` and a `user_prologue` in Section 3.2):
train `task="writing"` / `"twitter"`, `num_examples=-1`; eval `task="math"` / `"wildchat"`, `num_examples=50`.
The broad mixture also trains on `writing` with `max_sub_calls=64` and evaluates `congress` with `pool_size=200`.
