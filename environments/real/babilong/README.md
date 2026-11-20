# babilong-real

BABILong (bAbI facts hidden in PG19 text) as an `RLMTrainEnv` environment; the haystack is the `context` REPL
variable. Reward: official BABILong label match with whole-word matching (`metric="wordbound"`).

Data: HF `RMT-team/babilong` (`source="eval"`) and `RMT-team/babilong-train-5k-samples` (`source="train"`),
downloaded on first use (the loader lists repo files, so the first build needs network).

Arguments used by the paper (appendix broad mixture, `state_label_set=true`, `metric="wordbound"`):

| stream | `source` | `task` | `length` | size |
|---|---|---|---|---|
| train | `train` | `qa3` | `["32k","64k","128k"]` (no 128k in the train repo, skipped) | all rows |
| validation | `eval` | `qa1` | `128k` | `per_pair_examples=20` |
