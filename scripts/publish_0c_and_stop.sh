#!/usr/bin/env bash
# Waits for 0c scoring to finish, commits the small result parquets into the repo
# (so they're pullable without a live pod), then stops the pod.
set -u
cd /workspace/phrase-rl
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_POD_ID=')"
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_API_KEY=')"

while tmux has-session -t score0c 2>/dev/null; do sleep 60; done
echo "score0c ended; publishing results"

git pull --no-edit || true
mkdir -p results/phase0c/raw
cp -f data/rollouts_0c.parquet results/phase0c/raw/ 2>/dev/null || true
cp -f data/scores_0c.parquet   results/phase0c/raw/ 2>/dev/null || true
git add results/phase0c/raw
git -c user.name=pod -c user.email=pod@runpod commit -m "Phase 0c raw results (rollouts + flow-loss scores) [pod]" || true
git push || true

sleep 20
runpodctl stop pod "$RUNPOD_POD_ID"
