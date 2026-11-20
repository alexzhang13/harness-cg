# longreason-expanded

LongReason-style items built from reasoning MCQs that are not in LongReason (ReClor, LogiQA 2.0; near-duplicates of
LongReason originals removed): each passage was expanded by base Qwen3.8-27B into setting-restating paragraphs
(`scripts/gen_expanded.py`), which are scattered in order through DCLM-baseline filler at 128k (474k chars). Rows,
root prompt and scoring are `longreason_real`'s; sub-calls go to a real model.

Data, read at load time: `expanded_items.jsonl` (the 27B expansions used in the paper) and `filler.jsonl.gz`. Local
copies are used if present (`$LONGREASON_EXPANDED_ITEMS`, `$LONGREASON_FILLER`, or `longreason_expanded/data/`);
otherwise both are downloaded from HF `a1zhang/harness-cg-frozen-inputs` (`rlm_train.frozen`). The filler can be
rebuilt with `python -m longreason_expanded.build_filler --shard <dclm-baseline-1.0-parquet shard>`.

- RL (`rlstage-longreason`): `split="train", length="128k", max_iterations=20, sub_max_tokens=8192, min_subcall=0`.
- SFT trace generation: `split="train", num_examples=-1, max_iterations=20, sub_max_tokens=24000, min_subcall=0`.
- Splits: `train` = all items except every 10th (file order), `val` = every 10th.
