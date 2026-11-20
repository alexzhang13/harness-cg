# §4.1 RL on short tasks (Table main-composition "+ RL (short tasks)")

RL with the RLM harness (orchestrator prompt on) on the six short length-generalization streams (MRCRv2, GraphWalks,
LongBenchPro, OOLONG, OOLONG-Pairs, Ada-LEval), then evaluation on the held-out suite in `eval/`.

```bash
bash run_train.sh qwen3-30b-a3b    # LoRA, 1000 steps; the same run as appendix/short_task_mixture rlm
bash run_train.sh qwen38-27b       # full fine-tuning, lr 5e-6, 150 steps (8 GPUs: 6 train + 2 inference)
```

Paper checkpoints: Qwen3-30B-A3B step 1000 (LoRA adapter, served with `LORA=` in `training/serving/serve_vllm.sh`),
Qwen3.8-27B step 150. Full fine-tuning exports need key renaming before vLLM can load them:
`python training/prime_rl/tools/fix_export_names.py <weights/step_150> <Qwen3.8-27B snapshot> <out dir>`.

Evaluate any served model (also used by §4.2 and the broad-mixture appendix): `eval/README.md`.
