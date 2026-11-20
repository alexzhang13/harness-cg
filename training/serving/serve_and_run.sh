#!/usr/bin/env bash
# Serve a model (serve_vllm.sh + lb_proxy.py), run a command with BASE_URL and MODEL set, then stop the servers.
#
#   MODEL=<HF id or dir> SERVED=<name> [TOKENIZER=...] [TP=2] [MAX_MODEL_LEN=262144] [PORT=18431] \
#     bash training/serving/serve_and_run.sh <command ...>
#
# Other serve_vllm.sh variables pass through.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL:?}" "${SERVED:?}"
WORK="$(mktemp -d)"; export POOL="$WORK/pool.json"
PORT="${PORT:-18431}"; export PORT_BASE="${PORT_BASE:-$((PORT + 100))}"
(cd "$WORK" && bash "$HERE/serve_vllm.sh") > "$WORK/serve.log" 2>&1 &
SERVE=$!
trap 'kill $LB $SERVE 2>/dev/null; wait $SERVE 2>/dev/null; echo "server logs: $WORK"' EXIT
LB=""
until python3 -c "import json,sys; sys.exit(0 if json.load(open('$POOL'))['ports'] else 1)" 2>/dev/null; do
  kill -0 $SERVE 2>/dev/null || { echo "serve_vllm.sh exited; see $WORK" >&2; exit 1; }
  sleep 10
done
LB_POOL="$POOL" python "$HERE/lb_proxy.py" --listen "$PORT" > "$WORK/lb.log" 2>&1 &
LB=$!
sleep 5
export BASE_URL="http://127.0.0.1:$PORT/v1" MODEL="$SERVED"
"$@"
