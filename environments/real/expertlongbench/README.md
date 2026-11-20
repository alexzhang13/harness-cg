# expertlongbench-real

ExpertLongBench (`launch/ExpertLongBench`, auto-downloaded from the HF Hub) as an `RLMTrainEnv`. The model reads
the task input in `context` and fills the task's expert checklist (field names are given) as one JSON object.

- Tasks: `T01LegalMDS` (100 rows), `T03MaterialSEG` (50), `T04EduPAE` (100), `T06HealthCNG` (100); select with
  `task=`. Rows of the selected task(s) are shuffled with `seed` (default 42); ids are positions in that order.
- Reward: mean over checklist fields of an equivalence verdict against the human reference checklist. Fields that
  are N/A or identical are scored without a judge; the rest are graded in one call to `judge_model`
  (default `openai/gpt-4.1-mini`).
- Judge credentials are resolved by `rlm_train.judge_client` (`RLM_JUDGE_*`, `OPENAI_API_KEY`, or `PRIME_API_KEY`
  with `PRIME_API_TEAM`); the loader raises if none is set. A failed judge call sets `state["elb_judge_error"]`
  and scores the judged fields 0, so such rows must be re-judged.
- Held-out evaluation, once per task: `task="<task>", num_examples=-1, max_iterations=40, orchestrator_prompt=true,
  sub_extra_body={"chat_template_kwargs": {"enable_thinking": false}}`; ids 0-99 (0-49 for T03).
