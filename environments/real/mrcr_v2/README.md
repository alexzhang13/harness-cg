# mrcr-v2-real

MRCR v2 multi-needle conversation recall as an `RLMTrainEnv` environment. The transcript is the `context` REPL
variable; the model reproduces one earlier assistant response, prefixed with a given 12-character hash.
Reward: `difflib.SequenceMatcher` ratio on the text after the hash.

Data: public CSVs at `https://storage.googleapis.com/mrcr_v2/`, no account needed. With `num_examples > 0` the
first N rows are streamed; with `num_examples = -1` the file is downloaded to `~/.cache/rlm_train_real/mrcr_v2`.

Arguments used by the paper (`seed=42`, `max_iterations=20`):

| split | `needle_count` | `context_range` | `num_examples` |
|---|---|---|---|
| train | 2 | `32k-64k` | 512 |
| eval | 8 | `2m-4m` | 50 |
| eval, base Transformer (single-turn env) | 8 | `1m-2m` | 50 |
