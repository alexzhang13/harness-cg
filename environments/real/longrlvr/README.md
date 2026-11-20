# longrlvr-real

LongRLVR grounded long-document QA as an `RLMTrainEnv` environment. `context` is
`{"question", "chunks"}`; the model submits `{"answer", "useful_chunks"}`. Reward: upstream
`ans + 0.1*fbeta + 0.9*ans*fbeta`, divided by 2, with `ans` from a binary LLM judge.

Data: HF `Guanzheng/LongRLVR-Data` (parquet, downloaded on first use). The judge needs `RLM_JUDGE_API_KEY` /
`RLM_JUDGE_BASE_URL` (+ `RLM_JUDGE_MODEL`), `OPENAI_API_KEY` or `PRIME_API_KEY` (resolved by
`rlm_train.judge_client`; default model `openai/gpt-4.1-mini`).

Arguments used by the paper (`seed=42`, `answer_metric="judge"`):
- train: `subset="llama", split="train", length=["16k","32k"], num_examples=2000`
- validation: `subset="qwen2.5-7b", split="test", num_examples=20`
