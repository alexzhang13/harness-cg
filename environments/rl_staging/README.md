# rl-staging (`rlstage-*` env ids)

The RL training streams of Section 4.2: the eight synthetic environments in the form that produced the SFT data, with
every hint / note switched off (a hint-bearing argument raises unless `allow_hint=True`), plus the OOLONG-real
validation env. Each wrapper pins arguments and forwards to the package's loader.

| id | package | sub-LM | pinned arguments |
|---|---|---|---|
| `rlstage-bcp` | `browsecomp-plus-synth` | fake | `num_examples=2000, hint="none", reward_mode="outcome"` |
| `rlstage-ctc` | `ctcbench-synth` | fake | 8 families, `rungs="native", seed=7` |
| `rlstage-elb` | `expertlongbench_synth.mix` | fake | `num_examples=4000, p_docket=0.275, form={"p_recipe":0.345,"p_tutor":0.54}` |
| `rlstage-obliq` | `trick-synth` | fake | `read={"preset":"real_v1", faults 0.15}, max_iterations=20, max_sub_calls=400` |
| `rlstage-corpusqa` | `corpusqa-synth` | real | `family="periodic", surface="varied", sub_lm="real"`, `scale` "1m" / "128k" |
| `rlstage-loogle` | `loogle-strategy-synth` | real | `family="mix", note="", reward_mode="outcome", generator_version=2, max_iterations=30` |
| `rlstage-oolong` | `tabletop-synth` | real | `seed=100000` (train) / `900000` (val), `n_episodes=3, surfaces="v2"` |
| `rlstage-longreason` | `longreason-expanded` | real | `length="128k", max_iterations=20, sub_max_tokens=8192, min_subcall=0` |
| `rlstage-oolong-real-val` | `oolong-real-synth` | real | `path=$OOLONG_REAL_DATA_DIR/dnd/validation.jsonl`, 25 windows, `max_episodes=3, seed=0` |
