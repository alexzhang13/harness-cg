# Appendix A: short-task mixture (Table `generalization`)

Each method trains on an equal-weight mixture of the six short length-generalization splits (§3.1), then is evaluated
at temperature 0 on nine long or held-out-domain cells: the six long length-generalization splits, plus OOLONG
spam 132k, OBLIQ math and OBLIQ wildchat. The RLM run is also the Qwen3-30B-A3B "+ RL (short tasks)" row of Table
`main-composition`.

```bash
bash run_train.sh <rlm|lm_repl|pi|transformer>     # configs/<method>.toml
bash eval.sh <rlm|lm_repl|pi|transformer> [group]   # offline evaluation of the saved adapter, cells/<method>*.jsonl
```

`eval.sh` serves the adapter with `training/prime_rl/offline_eval/run_offline_cells.sh` and writes to
`$PRIME_RL_DIR/outputs/short-mix/<method>/offline_eval/`. Paper checkpoints: RLM and LM + REPL step 1000, Pi 300,
Transformer 420 (override with `STEPS=`; `EVAL_BASE=1` also scores the base model). The Transformer is evaluated in
groups, one server context window each (see the header of `eval.sh`). Its MRCRv2 cell uses the 1m-2m split, because
2m-4m does not fit.

The single-task rows of the table are the Fig. length-gen runs (`../../sec3.1_length_generalization/run_train.sh
rlm_orchestrator_prompt <task>`).

