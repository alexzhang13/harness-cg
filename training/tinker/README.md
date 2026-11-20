# Tinker backend

A [Tinker](https://tinker-docs.thinkingmachines.ai/) equivalent of the LoRA experiments. It reads the same config TOML
as `training/prime_rl/launch.sh`, runs the same environments and harness code, and reimplements prime-rl's
orchestrator and trainer (advantages, loss, filters, async batching) around Tinker's sampling and training calls.
Full fine-tuning runs (Qwen3.8-27B RL and all SFT) stay on prime-rl.

## Setup

```bash
uv venv --python 3.12 .venv-tinker
uv pip install --python .venv-tinker/bin/python -e . -e training/tinker
for d in $(find environments -name pyproject.toml -exec dirname {} \;); do
  uv pip install --python .venv-tinker/bin/python --no-deps -e "$d"; done
uv pip install --python .venv-tinker/bin/python python-dateutil pytrec-eval-terrier pandas
export TINKER_PYTHON=$PWD/.venv-tinker/bin/python TINKER_API_KEY=...
```

## Running

Every experiment launcher takes `BACKEND=tinker`:

```bash
BACKEND=tinker bash experiments/sec3.1_length_generalization/run_train.sh rlm_orchestrator_prompt oolong
bash training/tinker/launch.sh <config.toml>          # any config directly
```

Runs go to `$TINKER_RUNS_DIR/<output_dir>-seed<SEED>` (default `~/tinker-runs`) and resume from their last checkpoint.
`rlm-train-tinker eval <run_dir>` evaluates saved checkpoints on Tinker; `rlm-train-tinker export <run_dir>` downloads
them as PEFT adapters in prime-rl's layout, for the offline evaluators that need contexts above 262k tokens.
`rlm-train-tinker show <config>` prints the resolved config and the settings that differ from prime-rl.
