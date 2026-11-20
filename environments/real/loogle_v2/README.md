# loogle-v2-real

LooGLE-v2 (`MuLabPKU/LooGLE-v2`, test split, 1,934 rows; auto-downloaded from the HF Hub) as an `RLMTrainEnv`:
questions over one long real document (law, finance, code, game logs). `context` is the document; the root prompt
is the upstream instruction with the question and options.

- Evaluation cohort (the loader default, `indices="eval"`): the 206 rows whose document has at least 2,000,000
  characters (500K tokens at 4 characters per token), frozen as dataset positions in
  `loogle_v2_real/eval_indices.json`. The same rows result from `indices=None, min_context_tokens=500000`.
- `indices` also accepts a list of dataset positions (subsets of the cohort, e.g. to resume a run) or `None`
  (whole split); `source`, `task`, `answer_type` and the token-band filters apply after it.
- Scoring is deterministic (letter / identifier match, set-F1 for file lists, exact or tolerance numeric match).
- Held-out evaluation: `evaluation_version="2026-10-11", seed=None` (dataset order), `max_iterations=12`,
  `max_sub_calls=80`, `sub_extra_body={"chat_template_kwargs": {"enable_thinking": false}}`.
