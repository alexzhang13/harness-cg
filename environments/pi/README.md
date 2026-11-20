# Pi baselines (verifiers v1)

One package per benchmark, `pi_<x>_v1` (env id `pi-<x>-v1`): `oolong`, `oolong_pairs`, `ada_leval`, `graphwalks`,
`longbenchpro`, `mrcr_v2`, `obliq`. Each imports the dataset builder and grader of its backing env under
`environments/real/<x>` (installed, or loaded from this checkout by `rlm_train.pi.load_backing_env`) and wraps them
with `rlm_train.pi.make_pi_env`: the context goes to `context.txt` in pi's sandbox, the instruction is the shared
preamble + task + the env's `OUTPUT FORMAT` footer, and the reward is the backing grader on pi's output.

Data setup is that of the backing env. Arguments: the backing env's dataset arguments as in the RLM configs, plus
`max_turns` (30 in the paper) and further `HostPi` keyword arguments. Needs a verifiers v1 interpreter and node +
pi with the extensions (see `training/README.md`).
