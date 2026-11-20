# Pi on MRCRv2, LongBench-Pro, OOLONG-Pairs and Ada-LEval

The Pi baseline (coding agent with sub-agents) on these four tasks trains with hard limits and saves every LoRA checkpoint;
the checkpoints are evaluated afterwards on separate servers (`EVAL.md`). `../run_train.sh pi <task>` runs this setup.
GraphWalks and OOLONG use `../configs/pi/<task>.toml` like the other methods, with the same limits.

| | |
|---|---|
| Model | Qwen3-30B-A3B-Instruct-2507, LoRA rank 32, 150 steps, batch 64 x 4 rollouts, seed 0 |
| Sampling | T 0.7, top-p 0.8, top-k 20; 4,096 tokens per model call (OOLONG-Pairs: 12,288 train, 131,072 eval) |
| Window | 65,536 tokens in training, 262,144 in evaluation; no compaction |
| Node | 8 GPUs: 6 inference (tensor parallel 2 x data parallel 3) + 2 trainer |

## Pi limits (`settings.py`)

| Limit | Value |
|---|---|
| Root agent turns | 30 (hard: the 31st model call is never issued) |
| Sub-agent calls per rollout | 100 |
| Turns per sub-agent | 10 (hard) |
| Wall clock | root 600 s, sub-agent 120 s (evaluation: 1,800 s / 300 s; OOLONG-Pairs 3,600 s / 600 s) |
| Tool output | 6 KB per `read` (sub-agents on Ada-LEval and OOLONG-Pairs: 96 KB) |

Hitting a limit ends the rollout cleanly with no answer (reward 0), the same convention as an RLM reaching
`max_iterations`. The prompt (`PI_PROMPT_VARIANT=short`) quotes the limits from the same variables.

Task-specific settings (`TASK_PI_ENV`, `TASK_TRAIN` in `settings.py`):
- LongBench-Pro, Ada-LEval, OOLONG-Pairs: a prompt guide for reading large files (`wc`, `grep`, `sed -n`, the
  `reader` / `worker` helpers) and turn-budget notices over the last 6 turns (`PI_PROMPT_GUIDE`, `PI_TURN_WARN_BEFORE`).
- Ada-LEval, OOLONG-Pairs: a short task paragraph (`PI_PROMPT_TASK`) and trainer `seq_len` 32,768 (others: 8,192).

## Run layout

```
$PRIME_RL_DIR/outputs/length-gen/pi/<task>/seed0/
  config.toml settings.json code_rev.txt     what was run
  adapters/step_10 ... step_150/             LoRA checkpoints
  eval/                                      evaluation results (EVAL.md)
  COMPLETE                                   written when all 15 adapters exist
```

`run.sh` skips a complete run, resumes a partial one from its newest trainer checkpoint, and copies each adapter to
`adapters/` while training.

## Commands

```bash
bash deploy_pi_ext.sh               # deploy the sub-agent extension with the call budget (once per install)
bash run.sh <task> 0                # train on the current node (what ../run_train.sh pi <task> runs)
python submit.py --tasks mrcr       # or submit to SLURM
python status.py                    # one row per run
python verify_caps.py <run_dir>     # audit a run's rollouts against the limits
```
