#!/usr/bin/env bash
# Deploy the patched sub-agent extension into the pi package as `subagent-cg` (it imports pi internals).
# Re-run after changing extensions/subagent/index.patch.
#   bash deploy_pi_ext.sh
set -euo pipefail
HCG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PI_PKG="${PI_PKG:-$HOME/node/lib/node_modules/@mariozechner/pi-coding-agent}"
EXT_SRC="$PI_PKG/examples/extensions/subagent"
DEST="$PI_PKG/examples/extensions/subagent-cg"
[[ -f "$EXT_SRC/index.ts" ]] || { echo "pi example extension not found: $EXT_SRC" >&2; exit 1; }
tmp="$(mktemp -d)"
cp "$EXT_SRC/index.ts" "$EXT_SRC/agents.ts" "$tmp/"
patch -d "$tmp" -p1 < "$HCG/src/rlm_train/pi/extensions/subagent/index.patch"
grep -q "rlm-train baseline wiring" "$tmp/index.ts"
grep -q "PI_SUBAGENT_MAX_CALLS" "$tmp/index.ts"
grep -q "PI_TURNCAP_EXT" "$tmp/index.ts"
grep -q "PI_SUBAGENT_TOOL_OUTPUT_CAP_BYTES" "$tmp/index.ts"
"$HOME/node/bin/node" --experimental-strip-types --no-warnings --check "$tmp/index.ts"
# atomic replace: running jobs load this file on every rollout
mkdir -p "$DEST"
for f in agents.ts index.ts; do cp "$tmp/$f" "$DEST/$f.new" && mv -f "$DEST/$f.new" "$DEST/$f"; done
rm -rf "$tmp"
echo "deployed $DEST"
