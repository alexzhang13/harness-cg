# trick-synth (OBLIQ stand-in)

OBLIQ-shaped analogical retrieval over generated problems with a hidden core technique: the root ranks a pool of
~2,500-3,100 problems by whether they share the query's technique; reward nDCG@50. Sub-calls are answered by a fake
sub-LM calibrated to Qwen3.8-27B on OBLIQ math (`oracle.py`, preset `real_v1`, optional reply faults).

Corpus: `corpus.json`, 125 techniques and 5,054 problems written by base Qwen3.8-27B (thinking off) with
`tools/gen_corpus.py` -> `tools/rewrite_corpus.py` -> `tools/fix_corpus.py`. A local `trick_synth/data/corpus.json`
(or `$TRICK_CORPUS`, a file name under `data/` or an absolute path) is used if present; otherwise the paper's file is
downloaded from HF `a1zhang/harness-cg-frozen-inputs` (a rebuild samples a new corpus).

- RL (`rlstage-obliq`): `split="train", num_examples=2000, read={"preset": "real_v1", "fault_q": 0.15, "fault_desc":
  0.15, "fault_flag": 0.15}, max_iterations=20, max_sub_calls=400, hint="none", hint_prob=0.0`.
- SFT trace generation (`synth_common.probe`): `split="train", num_examples=10000`, the same `read`, and the program
  note as `user_prologue="<hint>...</hint>"`.
- Splits: 20% of techniques held out for `val`.
