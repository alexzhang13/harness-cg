# longbenchpro-real

LongBench-Pro as an `RLMTrainEnv` environment. The long document is the `context` REPL variable; the reward is
the benchmark's per-task metric chosen by the `secondary_task` prefix (nDCG for T1, pairwise accuracy for T2 and
T6.3, accuracy for T3 and T11, F1 for T5, T6.2, T7, T9, sub-EM for T6.1, T8, T10). Summarization (T4) is excluded.

Data: HF `caskcsg/LongBench-Pro`, `test` split (downloaded on first use). nDCG needs `pytrec-eval-terrier`.

Arguments used by the paper (`language="English"`, `thinking=false`, `seed=42`, `max_iterations=20`):

| split | `token_length` | `num_examples` |
|---|---|---|
| train | `32k` (broad mixture: `16k`, `32k`, `64k`) | -1 |
| eval | `256k` | -1 (115 rows) |
