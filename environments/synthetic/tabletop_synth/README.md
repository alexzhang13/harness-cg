# tabletop-synth (OOLONG-Real stand-in)

Synthetic tabletop (D&D) session logs: an invented cast, 1-3 episodes of 700-1,400 lines rendered from hidden roll /
spell event tables, and one OOLONG-real-style question with an exact gold (`generate.py`; vocabulary in `vocab.py`
plus `oolong_real_synth.world` / `gen`). No external data. Sub-calls go to the real model (no fake oracle); reward is
`oolong_real_synth.scoring.score_synth` (the paper metric).

- RL (`rlstage-oolong`): `num_examples=2000, seed=100000` (train) / `900000` (val), `n_episodes=3, surfaces="v2",
  max_iterations=20, max_sub_calls=600`.
- SFT trace generation: one row per seed (`num_examples=1, seed=<seed>`, optional `tid`), `surfaces="v1"` or `"v2"`,
  sub-calls thinking off, `default_answer_at_limit=False`.
