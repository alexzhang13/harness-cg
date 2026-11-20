# personamem-v2

PersonaMem-v2 implicit-persona personalization as an `RLMTrainEnv` environment. `context` holds the persona's chat
history; the root prompt holds the user query and four candidate replies. Reward: MCQ accuracy.

Data: HF `bowen-upenn/PersonaMem-v2` (`benchmark/text/*.csv` plus per-persona chat-history JSON, fetched on
demand; licence CC-BY-4.0).

Arguments used by the paper (`question_type="all"`, `seed=42`):
- train: `split="train", context_size="32k", num_examples=2000`
- validation: `split="benchmark", context_size="128k", num_examples=20`
