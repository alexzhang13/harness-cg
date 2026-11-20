# oolong-pairs-real

OOLONG-Pairs as an `RLMTrainEnv` environment: list every unordered pair of user ids whose hidden label histories
satisfy a condition, over one `trec_coarse` window of OOLONG-synth. 20 questions per context length, all on the
same context. Reward: F1 over `(id_lo, id_hi)` pair sets.

Data: questions + gold pairs are the published set HF `mit-oasys/oolong-pairs` (`data/oolong-pairs-<n>.json`).
The 4096 / 8192 / 16384 files are bundled in `oolong_pairs_real/data/`; 32768 is downloaded from that dataset
(pinned revision) on first use. The context comes from HF `oolongbench/oolong-synth` (`validation` split).

Arguments used by the paper (`seed=42`, `max_iterations=30`, `max_sub_calls=1000`):

| split | `context_length` | `num_examples` |
|---|---|---|
| train | 8192 (broad mixture: 4096 and 8192; `shuffle=true`) | -1 (20 questions) |
| eval | 32768 | 20 |
| validation of the base Transformer mixture run (single-turn env) | 16384 | 20 |
