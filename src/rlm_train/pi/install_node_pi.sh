#!/usr/bin/env bash
# Install node and the pi CLI into $NODE_DIR (default ~/node) and deploy the sub-agent and compaction extensions
# into the pi package (they import pi internals).
#   node 22.14.0, @mariozechner/pi-coding-agent 0.73.1
# Re-run after changing an extension.
set -euo pipefail

NODE_VERSION="${NODE_VERSION:-22.14.0}"
PI_VERSION="${PI_VERSION:-0.73.1}"
NODE_DIR="${NODE_DIR:-$HOME/node}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$("$NODE_DIR/bin/node" --version 2>/dev/null || true)" != "v$NODE_VERSION" ]]; then
  tmp="$(mktemp -d)"
  curl -sL "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz" -o "$tmp/node.tar.xz"
  rm -rf "$NODE_DIR"
  mkdir -p "$NODE_DIR"
  tar -xJf "$tmp/node.tar.xz" -C "$NODE_DIR" --strip-components=1
  rm -rf "$tmp"
fi
export PATH="$NODE_DIR/bin:$PATH"

npm install -g "@mariozechner/pi-coding-agent@$PI_VERSION"
PI_PKG="$NODE_DIR/lib/node_modules/@mariozechner/pi-coding-agent"
installed="$(node -e "console.log(require('$PI_PKG/package.json').version)")"
[[ "$installed" == "$PI_VERSION" ]] || { echo "pi version $installed != pinned $PI_VERSION" >&2; exit 1; }

# pi's sub-agent example, patched: children use the root's model, readcap and their own turn budget
SUB_DEST="$PI_PKG/examples/extensions/subagent-rlm"
rm -rf "$SUB_DEST"
mkdir -p "$SUB_DEST"
cp "$PI_PKG/examples/extensions/subagent/index.ts" "$PI_PKG/examples/extensions/subagent/agents.ts" "$SUB_DEST/"
patch -d "$SUB_DEST" -p1 < "$HERE/extensions/subagent/index.patch"
grep -q "rlm-train baseline wiring" "$SUB_DEST/index.ts"
grep -q "PI_SUBAGENT_MAX_CALLS" "$SUB_DEST/index.ts"

CMP_DEST="$PI_PKG/examples/extensions/compaction-rlm"
rm -rf "$CMP_DEST"
mkdir -p "$CMP_DEST"
cp "$HERE/extensions/compaction/index.ts" "$CMP_DEST/"

echo "installed node $(node --version), pi $installed; extensions deployed under $PI_PKG/examples/extensions"
