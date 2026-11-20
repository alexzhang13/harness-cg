# synth-common

Shared library of the synthetic training environments (no env id of its own).

- `envkit.py`: seed-only dataset rows; context, root prompt, gold and hidden facts generated per rollout in
  `setup_state`; `make_env` / `make_rubric` (outcome reward + `prog_*` / `synth_*` metrics).
- `oracle.py`: `FakeSubLM` / `ReadModel`, the base of every programmed sub-LM (read budget, misses, errors, refusals).
- `segindex.py`: which context items a sub-call prompt actually contained (coverage accounting).
- `program.py`: hint tiers (`strategy`, `program`, `verify`) appended as `<hint>...</hint>`, program checks and
  `reward_mode` (`outcome` | `gated` | `shaped`).
- `scripted.py`: run an env with a scripted root (tests, reference programs); `probe.py`: Prime Inference client.
