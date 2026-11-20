# ctcbench-synth

Synthetic CTC-Bench: real CTC `split_v1` train / val rows (from `ctcbench-real`, which needs the generator-built
`$CTCBENCH_TRAIN_DIR`), single or recombined toward the long eval rungs, rendered with the benchmark's own prompt and
scored by its own metric. Sub-calls are answered by gold-driven fake sub-LMs calibrated to the real 27B (`oracles.py`,
`grouping_real.py`); `real_sub_calls=True` sends them to the real sub-LM instead (thinking off), with the same rows,
prompts and scoring. Rows overlapping any published eval row are dropped or scrubbed (`decontam.py`).

Sidecars (`ctcbench_synth/sidecars/`; a missing one is an error): `reorder_seams.json` (shipped);
`outlier_articles.json.gz` (shipped, frozen; `tools/build_sidecars.py outlier` builds a new one from the decontaminated
rows);
`eval_index.npz` (`tools/build_eval_index.py`, hashes of every `PrasannSinghal/ctc-suite-eval` item, about 70 s);
`xabsence_para_pool.json.gz` (family `xabsence_paraphrase` only; `$XABSENCE_PARA_POOL` or downloaded from HF
`a1zhang/harness-cg-frozen-inputs`; written by a served 27B with `tools/build_xabsence_para_pool.py`, then
`tools/decontam_xabsence_pool.py`). `surface.py` renders rows in varied
formats (used by the SFT trace builders).

- RL (`rlstage-ctc`): `split="train", family="outlier,contradiction_iid,grouping,textgroups,reorder,qdmatch_hpqa,
  hotpotqa,nq", rungs="native", num_examples=2000, seed=7, hint="none", hint_prob=0.0, reward_mode="outcome"`.
