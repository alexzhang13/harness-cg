# expertlongbench-synth (`expertlongbench_synth.mix`)

Synthetic ExpertLongBench: fill an expert checklist from procedurally generated records. Docket rows (T01-like court
dockets, 108k-1.2M chars) and short forms rows (visit / tutor / recipe: T06 / T04 / T03-like) are interleaved; the
sub-LM is a programmed reader (omissions, misreadings, confabulations) and the reward is a deterministic per-field
matcher. No external data; no real ELB rows are used. No entry point of its own: the RL wrapper `rlstage-elb` loads
`expertlongbench_synth.mix.load_environment`.

- RL: `split="train", num_examples=4000, p_docket=0.275, form={"p_recipe": 0.345, "p_tutor": 0.54}, hint="none",
  reward_mode="outcome"`.
- SFT trace generation (per kind): `kind="docket"|"visit"|"tutor"|"recipe", hint="note", hint_prob=1.0` (each kind's
  final note), `orchestrator_prompt=True`, sub-calls thinking off.
- Splits: `train` / `val` / `test` seed ranges (disjoint).
