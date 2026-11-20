#!/usr/bin/env bash
# Install prime-rl at the pinned commit with this repo's patches, then install the harness and env packages
# into its environment.
#
#   PRIME_RL_DIR=~/prime-rl bash training/prime_rl/install.sh [env package dir ...]
#
# With no arguments every package under environments/ is installed. Re-running is safe.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
PRIME_RL_COMMIT="eaf95997f"
# some submodules use SSH URLs; fetch over HTTPS
HTTPS_SUBMODULES=(-c url."https://github.com/".insteadOf="git@github.com:")

if [[ ! -d "$PRIME_RL_DIR/.git" ]]; then
  git clone https://github.com/PrimeIntellect-ai/prime-rl.git "$PRIME_RL_DIR"
  git -C "$PRIME_RL_DIR" checkout "$PRIME_RL_COMMIT"
  git -C "$PRIME_RL_DIR" "${HTTPS_SUBMODULES[@]}" submodule update --init deps/verifiers deps/renderers deps/pydantic-config deps/research-environments
  for p in "$REPO"/training/prime_rl/patches/0*.patch; do
    git -C "$PRIME_RL_DIR" apply "$p"
  done
  for p in "$REPO"/training/prime_rl/patches/verifiers-*.patch; do
    git -C "$PRIME_RL_DIR/deps/verifiers" apply "$p"
  done
  (cd "$PRIME_RL_DIR" && uv sync --all-extras)
fi

# Optional second checkout for the §4.3 gated SFT runs (patches/sft_gated/).
if [[ -n "${PRIME_RL_SFT_GATED_DIR:-}" && ! -d "$PRIME_RL_SFT_GATED_DIR/.git" ]]; then
  git clone https://github.com/PrimeIntellect-ai/prime-rl.git "$PRIME_RL_SFT_GATED_DIR"
  git -C "$PRIME_RL_SFT_GATED_DIR" checkout "$PRIME_RL_COMMIT"
  git -C "$PRIME_RL_SFT_GATED_DIR" "${HTTPS_SUBMODULES[@]}" submodule update --init deps/verifiers deps/renderers deps/pydantic-config deps/research-environments
  for p in "$REPO"/training/prime_rl/patches/sft_gated/*.patch; do
    git -C "$PRIME_RL_SFT_GATED_DIR" apply "$p"
  done
fi

cd "$PRIME_RL_DIR"
uv pip install -e "$REPO"
if [[ $# -gt 0 ]]; then
  PKGS=("$@")
else
  mapfile -t PKGS < <(find "$REPO/environments" -name pyproject.toml -not -path '*/node_modules/*' -exec dirname {} \; | sort)
fi
for pkg in "${PKGS[@]}"; do
  uv pip install --no-deps -e "$pkg"
done
echo "installed rlm-train and ${#PKGS[@]} env package(s) into $PRIME_RL_DIR"
