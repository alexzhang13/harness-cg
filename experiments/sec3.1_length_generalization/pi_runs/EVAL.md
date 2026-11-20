# Checkpoint evaluation (Pi)

Intermediate checkpoints are evaluated while training runs, on separate 2-GPU servers (SLURM, 2 GPUs per server).
Each job (`slurm/eval_server.sbatch`) starts vLLM for Qwen3-30B-A3B-Instruct-2507 (tensor parallel 2, LoRA enabled with
runtime loading, `--max-model-len 262144` (the full window), hermes tool parser) and runs `../eval_pi_checkpoints.py --watch` on
`<run>/adapters`. Output goes to `<run>/eval/`:

```
<run>/eval/results/step_N/shard_<i>.jsonl          one rollout record per line (reward, metrics, error, stop_condition)
<run>/eval/sidecar/step_N/shard_<i>.jsonl.gz       trajectories (token arrays replaced by their lengths)
<run>/eval/sidecar/step_N/shard_<i>.manifest.json  host, job, sampling, env arguments and every PI_* variable
<run>/eval/claims/<ckpt>__<shard>/                 work claims shared by all servers of the run
<run>/eval/logs/slurm-<job>.out, vllm_logs/        job and vLLM logs
```

Tasks: `mrcr` (50 examples), `longbenchpro` (115), `oolong-pairs` (20), `adaleval` (50). The driver's unit is 10
examples, so a checkpoint is fully evaluated when `ceil(n / 10)` shards exist. Step 0 is the base model.

## Pi limits during evaluation

Those of training (`settings.py` `PI_ENV`, exported by `env.sh`): 30 root turns, 10 turns per sub-agent, 100 sub-agent
calls, short prompt, read cap on, compaction off. The wall clocks are longer and the window is the full 262,144 tokens (training: 65,536), set in the sbatch after
sourcing `env.sh`: `PI_WALL_TIMEOUT_S=1800`, `PI_SUBAGENT_WALL_TIMEOUT_S=300`, `PI_COMMAND_TIMEOUT=1860`,
`PI_ROLLOUT_TIMEOUT=1920`. The values in force are in each shard's `manifest.json` under `pi_env`.

## Commands

```bash
cd experiments/sec3.1_length_generalization/pi_runs
python submit_eval.py --dry-run                              # plan: runs, adapters, units left, jobs alive
python submit_eval.py --servers-per-run 2 --max-jobs 6       # submit; re-run (cron or loop) to keep servers alive
python submit_eval.py --tasks mrcr --seeds 0 --exit-after-idle 3600
python eval_status.py [--curves] [--tasks ..] [--seeds ..]   # adapters, evaluated checkpoints, mean reward, errors, cap hits

# one job by hand
mkdir -p <run>/eval/logs
sbatch -J cgpi-eval-<task>-s<seed> -o <run>/eval/logs/slurm-%j.out \
       --export=ALL,RUN=<run>,TASK=<task> slurm/eval_server.sbatch
```

`submit_eval.py` keeps up to `--servers-per-run` (default 2, never more than the units left) jobs named
`cgpi-eval-<task>-s<seed>` pending or running per run, at most `--max-jobs` in total, and skips fully evaluated runs.
A job exits after `EXIT_AFTER_IDLE` seconds (default 3600) without claimable work, for example while waiting for the
next adapter; re-running `submit_eval.py` starts a new one when new adapters appear. Jobs are `--requeue`; a unit
claimed by a job that is no longer queued is taken over by the next server. `sbatch -o` needs the log directory to
exist beforehand (`submit_eval.py` creates it).

Extra driver arguments for tests: `EXTRA_DRIVER_ARGS="--n-examples 1"` (or `--only step_10:0`).
