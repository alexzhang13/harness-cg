# singleturn-baseline

The base Transformer baseline: the model gets the whole context in one prompt and answers in one turn (no REPL,
no sub-LLM). `task` picks the benchmark (`oolong`, `oolong_pairs`, `graphwalks`, `longbenchpro`, `obliq`,
`ada_leval_bestanswer`, `mrcr`); the dataset builder and grader are imported from the backing env package (all
seven are dependencies), so data and scoring match the harness runs. Data setup is that of the backing env.

Arguments: the backing env's dataset arguments as in the harness configs (e.g. `min_ctx` / `max_ctx`,
`context_length`, `split`, `token_length`, `setting` / `manifest_split`, `needle_count` / `context_range`;
OBLIQ's sub-task goes in `obliq_task`), plus `num_examples`, `seed`, `system_prompt` (overrides the default
system prompt; used by the OOLONG answer-only evaluation), and for `graphwalks` `max_prompt_chars=120000`
(training) and `skip_examples` (the base Transformer mixture validation set). Harness-only arguments are ignored.
Serve the model with a window that covers the prompt (YaRN for the 256k+ evaluations; see `training/README.md`).
