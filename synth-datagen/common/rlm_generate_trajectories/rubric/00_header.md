You are reviewing a trajectory produced by a Recursive Language Model (RLM): a root model that
holds a long `context` in a Python REPL, can call a sub-model with `llm_query(prompt)` /
`llm_query_batched(prompts)`, and submits via `answer["content"]`. We are selecting trajectories
to fine-tune a model on, so we want ones a student should IMITATE.

Judge the PROCESS, not just the answer. Criteria:
