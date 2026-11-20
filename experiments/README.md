# Experiments

One directory per paper section, each with the figure it produces, its configs, run scripts and a README with the
commands. Training goes through `run_train.sh` (prime-rl by default, `BACKEND=tinker` for Tinker); runs are written
under `$PRIME_RL_DIR/outputs/`.

| Paper section |
|---|
| [§3.1 length generalization (Fig. length-gen)](sec3.1_length_generalization) |
| [§3.2 strategy generalization (Fig. strategy-gen)](sec3.2_strategy_generalization) |
| [§3.3 trajectory similarity (Fig. jaccard, App. D)](sec3.3_lid_trajectories) |
| [§4.1 RL on short tasks, held-out evaluation suite](sec4.1_rl_on_short_tasks) |
| [§4.2 synthetic environments (SFT and RL)](sec4.2_synthetic_environments) |
| [§4.3 composition over gated task chains](sec4.3_recursion_composition) |
| [App. A short-task mixture (Table generalization)](appendix/short_task_mixture) |
| [App. A broad task mixture (Fig. broad-mixture-rl)](appendix/broad_mixture) |

Methods (`<method>`): `rlm`, `rlm_orchestrator_prompt`, `lm_repl`, `pi`, `transformer`; how each is implemented:
[`training/README.md`](../training/README.md).
