#!/usr/bin/env bash
# Self-stop: waits for the given tmux session (default 'score') to end, then
# stops this pod. Results live on /workspace, which survives pod stop.
set -u
SESSION="${1:-score}"
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_POD_ID=')"
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_API_KEY=')"

while tmux has-session -t "$SESSION" 2>/dev/null; do sleep 120; done
sleep 30
echo "score session ended; stopping pod $RUNPOD_POD_ID"
runpodctl stop pod "$RUNPOD_POD_ID"
