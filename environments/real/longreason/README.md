# longreason-real

LongReason (`lz1bytedance/LongReason`, auto-downloaded from the HF Hub) as an `RLMTrainEnv`: 794 multiple-choice
reasoning questions whose scenario is embedded in filler text at a target length. `context` is the background
passage, the root prompt is the question with its options, and the reward is exact option-letter match.

- `split` is the upstream length split; the evaluation uses `split="128k"` (the default), all 794 questions.
  The other upstream splits (`8k` ... `64k`, `original`, `expanded`) load through the same code.
- Rows are shuffled with `seed` (default 42); ids are positions in that order.
- Held-out evaluation: `{"split": "128k", "max_iterations": 40}`, optionally with
  `"sub_max_tokens": 24576, "sub_extra_body": {"chat_template_kwargs": {"enable_thinking": true}}` (thinking
  sub-calls) or `"sub_extra_body": {"chat_template_kwargs": {"enable_thinking": false}}`.
