# graphwalks-real

GraphWalks (OpenAI) as an `RLMTrainEnv` environment. Each example is a directed-graph edge list plus one
operation: BFS from a node to an exact depth, or the parents of a node. The full prompt is the `context` REPL
variable. Reward: set-F1 between the submitted node set and the gold `answer_nodes`.

Data: HF `openai/graphwalks` (two parquet shards, downloaded on first use).

Arguments used by the paper (`seed=42`, `max_iterations=20`):

| split | `split` | rows in shard | `num_examples` |
|---|---|---|---|
| train | `128k_and_shorter` | 750 | 512 |
| eval | `256k_to_1mil` | 400 | 50 |
