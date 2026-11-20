# longproc-real

LongProc theory-of-mind tracking as an `RLMTrainEnv` environment. The upstream prompt is the `context` REPL
variable and the model writes the belief trace. With `tom_dense_reward=true` the reward is per-line positional
accuracy; the strict upstream prefix metric is logged alongside (`tom_strict`, `tom_dense`).

Data: HF `PrincetonPLI/LongProc`, configs `tom_tracking_2k` / `tom_tracking_8k` (`test` split, 200 rows each).

Arguments used by the paper (train only): `tasks="tom_tracking", length="2k"` and `length="8k"`,
`tom_dense_reward=true, num_examples=-1`.
