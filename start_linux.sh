#!/usr/bin/env bash
# Tune-O-Clip on Linux (after ./setup_linux.sh).
#   ./start_linux.sh            the 24/7 bot: every 15 min poll the streamers in workspace/twitch_channels.txt, clip,
#                               judge and edit into shorts (ready_to_post/); raw clips are pruned 2 days after their
#                               short exists; harvesting pauses under 5 GB free. Extra args go to queue_worker.py,
#                               e.g. ./start_linux.sh --auto-post --post-platform youtube
#   ./start_linux.sh studio     the calibration studio web UI on http://127.0.0.1:8787 (this machine only)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "No .venv yet: run ./setup_linux.sh first"; exit 1; }
if [ "${1:-}" = "studio" ]; then
  shift
  exec "$PY" -u calibration_studio/review_server.py --no-browser "$@"
fi
exec "$PY" -u backend/queue_worker.py --daemon --poll-twitch-channels workspace/twitch_channels.txt \
  --auto-edit --prune-raw-days 2 "$@"
