#!/usr/bin/env bash
# Validate every generated config with prime-rl's own parser (no GPU needed), against a scratch output dir.
#   bash dry_run.sh [task ...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
RL="${PRIME_RL_PYTHON:+$(dirname "$PRIME_RL_PYTHON")/rl}"
RL="${RL:-$PRIME_RL_DIR/.venv/bin/rl}"
[[ -x "$RL" ]] || { echo "rl entrypoint not found: $RL (set PRIME_RL_PYTHON)" >&2; exit 1; }
TASKS=("$@"); [[ ${#TASKS[@]} -gt 0 ]] || TASKS=(mrcr longbenchpro oolong-pairs adaleval)
fail=0
cd "$PRIME_RL_DIR"
for t in "${TASKS[@]}"; do
  for cfg in "$HERE"/configs/"$t"/seed*.toml; do
    scratch="$(mktemp -d)"
    if "$RL" @ "$cfg" --output-dir "$scratch" --dry-run >"$scratch/out.log" 2>&1; then
      echo "ok    $t/$(basename "$cfg")"
    else
      echo "FAIL  $t/$(basename "$cfg")"; tail -5 "$scratch/out.log"; fail=1
    fi
    rm -rf "$scratch"
  done
done
exit $fail
