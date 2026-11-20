# nocha-real

NoCha book-length TRUE/FALSE claim verification as an `RLMTrainEnv` environment; the book is the `context` REPL
variable and the model submits `TRUE` or `FALSE`.

Data: the public classics sample of `marzenakrp/nocha`, downloaded from GitHub raw on first use into
`~/.cache/rlm_train_real/nocha`.

Arguments used by the paper (validation stream only): `num_examples=20, seed=42, max_iterations=12,
min_subcall=1`.
