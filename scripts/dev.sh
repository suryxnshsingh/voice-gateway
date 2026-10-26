#!/usr/bin/env bash
# Run the model worker and the gateway together; Ctrl-C stops both.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

# On a Mac the idle model gets swapped out under memory pressure; keep it warm.
[ "$(uname)" = "Darwin" ] && export VG_WORKER_KEEP_WARM_S="${VG_WORKER_KEEP_WARM_S:-45}"
(cd "$root/worker" && VG_WORKER_DATA_DIR=./data uv run vgworker) &
worker=$!
trap 'kill $worker 2>/dev/null' EXIT

echo "waiting for the worker to load its models (about a minute on first start)..."
until curl -sf http://127.0.0.1:8100/health >/dev/null; do
  kill -0 $worker 2>/dev/null || { echo "worker exited" >&2; exit 1; }
  sleep 1
done

echo "playground: http://127.0.0.1:8000/playground"
cd "$root/gateway" && uv run voicegw serve
