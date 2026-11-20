# Training

The harness code is in `src/rlm_train/` and is trainer-agnostic: every method is a
[verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment. This directory holds the trainer
integration. All paper runs used [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) (GRPO; trainer,
vLLM inference server and orchestrator on one node). The LoRA runs can also be trained on Tinker: `tinker/README.md`.

## Methods

| Method | Environment | What is trained |
|---|---|---|
| RLM | `rlm_train.RLMTrainEnv` | root turns only |
| RLM + orchestrator prompt | `RLMTrainEnv(orchestrator_prompt=True)` | root turns only |
| LM + REPL | `RLMTrainEnv(enable_sub_llm=False)` | every turn (there are no sub-calls) |
| Pi | `rlm_train.pi.make_pi_env` (verifiers v1 `vf.Pi`) | root and sub-agent turns |
| Base Transformer | `environments/singleturn` (`singleturn-baseline`) | the single completion |

**RLM** (`src/rlm_train/env.py`). `rlm.RLM.completion` at depth 1. The context is a variable in a persistent Python
REPL (one subprocess per rollout, `worker.py`); the root model writes ```` ```repl ```` blocks; `llm_query` /
`llm_query_batched` in the REPL call the same served model through a localhost proxy (`proxy.py`). REPL output
is truncated to 20,000 characters per block. The rollout ends when a block sets `answer["ready"] = True`. Each
root turn is a trajectory step; sub-calls are logged as metrics and get no gradient. Prompts are in
`prompts.py`: `RLM_SYSTEM_PROMPT`, a metadata user message (query, context type and length), then `Turn i/N:`.

**RLM + orchestrator prompt.** Same, with `ORCHESTRATOR_PROMPT` appended to the system prompt.

**LM + REPL** (CodeAct-style, depth 0). Same loop with `enable_sub_llm=False`: the worker binds no sub-LLM
functions, the system prompt is `CODE_ONLY_SYSTEM_PROMPT`, the metadata message drops the sub-LLM sentence and
REPL output is not truncated.

**Pi** (`src/rlm_train/pi/`). The `pi` coding agent (0.73.1, Node 22.14.0) run by verifiers' v1 `vf.Pi` harness on a
host-process sandbox (`process_sandbox.py`). `HostPi` (`host_pi.py`) writes the context to `context.txt` in the
sandbox, passes the task as the prompt, and loads three extensions: `readcap` (6,000-byte tool-output cap and a
30-turn cap), `subagent` (a `subagent` tool with `reader` / `worker` helpers on the same model; 10 turns each)
and `compaction` (only when `PI_COMPACT_AT_TOKENS` is set). All model requests, root and sub-agent, pass through
verifiers' interception endpoint and become trajectory steps. `v1_env.py` turns a benchmark's dataset builder and
grader into a Pi env; the per-benchmark wrappers are in `environments/pi/`. Install node, pi and the extensions
with `bash src/rlm_train/pi/install_node_pi.sh`.

**Base Transformer + YaRN** (`environments/singleturn`). One completion: system prompt, then the full context and
the question as the user message. It reuses each benchmark's dataset builder and grader. Training uses a window
that fits the short split. Long evals that exceed the training-time server window are run offline on the saved
LoRA adapters (`prime_rl/offline_eval/run_offline_lora.sh`, temperature 0); where the eval split exceeds the
model's 262,144-token window the server is started with YaRN: `rope_scaling = {rope_type: "yarn", factor:
max(1, max_model_len / 262144), original_max_position_embeddings: 262144}`.

## prime-rl

```bash
PRIME_RL_DIR=~/prime-rl bash training/prime_rl/install.sh      # prime-rl @ eaf95997f + patches, harness, env packages
PRIME_RL_DIR=~/prime-rl bash training/prime_rl/launch.sh <config.toml>
```

`install.sh` pins prime-rl to `eaf95997f` (verifiers `v0.1.15.dev7`, renderers `v0.1.8.dev4`, vLLM 0.21) and applies
`prime_rl/patches/`:

1. `0001` balanced micro-batch packing across data-parallel trainer ranks.
2. `0002` log every rubric metric of eval rollouts as `eval/<env>/<metric>/avg@k` (the OBLIQ curves plot nDCG@10
   while training on nDCG@50).
3. `0003` checkpoint cleanup tolerates filesystem errors.
4. `verifiers-0001` (applied to `deps/verifiers`): with `RLM_MULTI_USER_TAIL=1`, the token-in/token-out renderer
   client extends the previous turn when the new messages are several user messages (REPL output, then the turn
   prompt), so a rollout trains as one sequence. Set by the Qwen3.8-27B RL scripts (`use_renderer = true`).

Configs live with the experiment that uses them (`experiments/*/configs/`). Shared settings of the RL runs:
GRPO, AdamW lr 5e-5 (constant), LoRA rank 32 / alpha 64 (Qwen3-30B-A3B; the adapters cover the attention projections, since the MLP names in
`target_modules` match no module of the MoE experts) or full
fine-tuning at lr 5e-6 (Qwen3.8-27B), batch 64 = 16 examples x 4 rollouts, `seq_len` 8192, bf16, 4 training and
4 inference GPUs. `launch.sh` exports `VLLM_USE_FLASHINFER_SAMPLER=0` and resumes from the latest checkpoint when
the run directory has one. `[inference.parallel]` (tensor / data parallel split of the inference GPUs) can be
changed to suit the node.

Saved adapters are under `<run>/run_default/broadcasts/step_N`. Evaluate adapters by serving them with
`--enable-lora` (as `offline_eval/run_offline_lora.sh` does); prime-rl's merged `weights/step_N` export of a LoRA
run does not contain the adapter.

## Supervised fine-tuning

`training/sft/` builds per-turn SFT rows from RLM traces and trains with prime-rl's `sft` entry point (used for
RLM-27B and for the gated-chain models). See `training/sft/README.md`.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `RLM_TRAIN_EXEC_TIMEOUT_S` | 600 | per-block REPL execution timeout |
| `RLM_TRAIN_REPL_REQUEST_TIMEOUT_S` | 1200 | host-side watchdog on one REPL request (kills the worker) |
| `RLM_TRAIN_SUBCALL_TIMEOUT_S` | 0 (proxy) / 600 (worker) | per-sub-call timeout |
| `RLM_TRAIN_SUB_CONCURRENCY` | 16 | concurrent sub-calls per rollout |
| `RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S` | 120 | worker start handshake |
| `RLM_TRAIN_REPL_MAX_MEM_GB` | 64 | memory cap of one REPL worker (0 disables) |
| `RLM_TRAIN_PROXY_MAX_BODY` | 256 MiB | largest sub-call request body |
| `RLM_DOC_STORE_LOCAL_ROOT` | `/tmp/rlm_doc_stores` | node-local mirror of corpus-scale contexts |
| `PI_*`, `VF_LOCAL_SANDBOX_*` | — | Pi baseline; see `src/rlm_train/pi/host_pi.py` and `process_sandbox.py` |
