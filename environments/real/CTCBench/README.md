# ctcbench-real (and `ctc-core-env`)

CTC-Bench (HF `PrasannSinghal/ctc-suite-eval`) as `RLMTrainEnv`s; prompt, parser and scorer are the benchmark's own
(`ctcbench_real/_vendor/ctc`, vendored verbatim from olmo-eval, Apache-2.0).

- `ctcbench-real`: `task=` one or more of the 22 suite rows; `split="eval"` (published ladders, by `rungs`), or
  `"train"` / `"val"` (generator-built rows under `$CTCBENCH_TRAIN_DIR`, default `~/ctcbench_data/train`).
- `ctc-core-env` (= `ctcbench_real.core.load_environment`): cells frozen in `core_v1.json`. Eval: `{"cells":
  "<task>@<rung>", "part": "test", "max_iterations": 30}` for `qdmatch_hpqa@r256k`, `contradiction_iid@r1m`,
  `outlier@r1m`, `xabsence_paraphrase@r32k`.
- Grouping eval: `task="grouping", split="eval", rungs="r32k", ids="grouping32_test", max_iterations=30, orchestrator_prompt=true`
  (60 rows frozen in `grouping32_v1.json`; rows keep their `example_id` in the full seed-42 order).
- RL validation: `task="outlier,contradiction_iid,grouping,reorder,hotpotqa", split="val", split_version="v1",
  per_task_examples=5, seed=7, max_iterations=30` (`split_v1.json`, built by `build_split_v1.py`).
- Train build (for `train` / `val`): `bash build_train.sh`, `N=1500 bash build_train.sh fiqa`, `bash build_extra.sh`,
  then `python merge_extra.py nq hotpotqa strmatch rerank xabsence absence textgroups fiqa qdmatch_nq` (row counts
  are checked against `split_v1.json`).
