#!/usr/bin/env bash
# Self-stop: waits for the 'score' tmux session to end, then stops this pod.
# Results live on the /workspace network volume, which survives pod stop.
set -u
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_POD_ID=')"
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_API_KEY=')"

while tmux has-session -t score 2>/dev/null; do sleep 120; done
sleep 30
echo "score session ended; stopping pod $RUNPOD_POD_ID"
runpodctl stop pod "$RUNPOD_POD_ID"
