# Source on a compute node. Selects the prime-rl environment, puts this repository's
# packages first on PYTHONPATH, and exports the Pi harness limits from settings.py.
HCG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PI_RUNS="$HCG/experiments/sec3.1_length_generalization/pi_runs"
export PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
if [[ -n "${ENV_TARBALL:-}" ]]; then   # unpack a prime-rl venv tarball to local disk once per node
  LOCAL_ENV="${CG_LOCAL_ENV:-${TMPDIR:-/tmp}/prime-rl-env-$USER}"
  (
    flock 9
    if [[ ! -f "$LOCAL_ENV/.stage-complete" ]]; then
      echo "[env] staging $ENV_TARBALL -> $LOCAL_ENV"
      rm -rf "$LOCAL_ENV"; mkdir -p "$LOCAL_ENV"
      zstd -d -q -c "$ENV_TARBALL" | tar -C "$LOCAL_ENV" -xf - && touch "$LOCAL_ENV/.stage-complete"
    fi
  ) 9>"${TMPDIR:-/tmp}/prime-rl-env-stage-$USER.lock"
else
  LOCAL_ENV="$PRIME_RL_DIR/.venv"
fi
export PRIME_RL_PYTHON="$LOCAL_ENV/bin/python" PATH="$LOCAL_ENV/bin:$PATH" VIRTUAL_ENV="$LOCAL_ENV"
RL="$LOCAL_ENV/bin/rl"

PP="$HCG/src"
while IFS= read -r d; do PP="$PP:$d"; done < <(find "$HCG/environments" -name pyproject.toml -not -path '*/node_modules/*' -exec dirname {} \; | sort)
export PYTHONPATH="$PP${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export ADALEVAL_DATA_DIR="${ADALEVAL_DATA_DIR:-$HOME/ada_leval_data}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RENDERERS=0.1.8
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-900}"
export NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=13
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_$USER}"; mkdir -p "$TRITON_CACHE_DIR"
export WANDB_MODE="${WANDB_MODE:-offline}"

# node, pi and the deployed sub-agent extension (deploy_pi_ext.sh)
export PI_NODE_BIN="${PI_NODE_BIN:-$HOME/node/bin}"
export PATH="$PI_NODE_BIN:$PATH" VF_LOCAL_SANDBOX_NODE_BIN="${VF_LOCAL_SANDBOX_NODE_BIN:-$PI_NODE_BIN}"
PI_PKG="$HOME/node/lib/node_modules/@mariozechner/pi-coding-agent"
export PI_SUBAGENT_EXT="${PI_SUBAGENT_EXT:-$PI_PKG/examples/extensions/subagent-cg/index.ts}"
[[ -f "$PI_SUBAGENT_EXT" ]] || { echo "[env] sub-agent extension not deployed: run deploy_pi_ext.sh" >&2; return 1 2>/dev/null || exit 1; }
grep -q PI_SUBAGENT_MAX_CALLS "$PI_SUBAGENT_EXT" || { echo "[env] deployed sub-agent extension lacks the call budget: rerun deploy_pi_ext.sh" >&2; return 1 2>/dev/null || exit 1; }

if [[ -f "$PRIME_RL_DIR/.env" ]]; then set -a; source "$PRIME_RL_DIR/.env"; set +a; fi
eval "$("$PRIME_RL_PYTHON" "$PI_RUNS/settings.py" env "${TASK:-}")"  # per-task additions for $TASK
