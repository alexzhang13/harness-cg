# manyih-priv

ManyIH-Bench instruction following, split by the number of `[[Privilege N]]` tags per query (train: at most
`train_max_tags`, eval: at least `eval_min_tags`). Reward: Constraint Success Rate from ManyIH's own checkers
(code checks plus an LLM judge); ISR logged as a metric.

Data: HF `jhu-clsp/ManyIH-Bench`, config `instruction_following` (if the dataset is gated for your account,
accept its terms on the Hub and set `HF_TOKEN`). The LLM judge is resolved by `rlm_train.judge_client` from
`RLM_JUDGE_BASE_URL` / `RLM_JUDGE_MODEL` / `RLM_JUDGE_API_KEY`; without them it calls the policy endpoint with
the policy model name (which fails on providers that do not serve that name, leaving only code checks).

Arguments used by the paper (`include_env_tips=true`, `seed=42`):
- train: `split="train", train_max_tags=5, max_examples=-1` (81 rows)
- validation: `split="eval", eval_min_tags=15, max_examples=20`
