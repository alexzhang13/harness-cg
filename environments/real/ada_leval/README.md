# ada-leval-real

Ada-LEval BestAnswer (StackSelect) as an `RLMTrainEnv` environment: a Stack Overflow question plus candidate
answers `A1..AN` (the accepted answer and answers to other questions); the model returns the accepted answer's
label. Reward: exact label match. `setting` (`8k`, `16k`, `32k`, `128k`) is the length of the candidate list.

Data: `stackselect_<setting>.json` from the Ada-LEval release (open-compass/Ada-LEval). The loader downloads it
on first use from `http://opencompass.openxlab.space/utils/AdaLEval/stackselect_<setting>.json` into
`$ADALEVAL_DATA_DIR` (default `~/.cache/rlm_train_real/ada_leval`); you can also place the files there yourself.
Question-disjoint splits (`manifest_split`) come from the packaged `bestanswer_disjoint_manifest.json`
(`train` 798 ids, `eval` 50, `train_long` 149); `python -m ada_leval_real.build_bestanswer_disjoint_manifest
--check` rebuilds it from `stackselect_{2k,32k}.json` and compares.

Arguments used by the paper (`task="bestanswer"`, `max_iterations=20`, `min_iterations=2`): train `setting="8k",
manifest_split="train", num_examples=-1`; eval `setting="128k", manifest_split="eval", num_examples=50`; broad
mixture also `16k` / `train` and `32k` / `train_long`. The RLM run with the orchestrator prompt adds a `user_prologue`.
