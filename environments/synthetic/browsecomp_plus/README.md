# browsecomp-plus-synth

Synthetic BrowseComp-Plus: find one entity in a procedurally generated ~27k-document encyclopedia (random knowledge
graph of regions, towns, organisations, people and works; near-twin decoys). No external data: each world is built
in a few seconds and written to a doc store under `$BCP_SYNTH_STORE_ROOT` (default `$RLM_DOC_STORE_LOCAL_ROOT`, else
`/tmp/rlm_doc_stores`). Sub-calls are answered by a programmed graph oracle (`BCPOracle`: read window, per-prompt
document cap, judgement errors). Reward: the real env's part-wise substring match, zero for hedged answers.

- RL (`rlstage-bcp`): `split="train", num_examples=2000, hint="none", hint_prob=0.0, reward_mode="outcome"`.
- SFT trace generation: `split="train", num_examples=4000, scale="full", max_iterations=30, max_sub_calls=120`
  (the commit stream adds `max_iterations=15, p_rare=0.0`).
- Splits: `train` (worlds `train0`, `train1`), `val` (held-out world, disjoint names and facts).
