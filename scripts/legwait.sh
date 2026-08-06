#!/bin/bash
# Waits for the v10 cell backlog to finish (26 cells) AND the local roll to be
# idle, then retires the consumer session and hands the pod to the leg queue.
cd /workspace/phrase-rl
while :; do
  timeout 120 git pull -q 2>/dev/null
  n=$(ls results/analysis/v10cells/*.json 2>/dev/null | wc -l)
  if [ "$n" -ge 28 ] && ! pgrep -f "[p]hase0c_rollout" >/dev/null 2>&1; then
    tmux kill-session -t consumer 2>/dev/null
    echo "[legwait $(date +%H:%M)] cells=$n -> leg mode" >> /workspace/leg.log
    exec bash scripts/claim_and_run_leg.sh
  fi
  sleep 300
done
