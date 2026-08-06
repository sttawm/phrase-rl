#!/bin/bash
# Claim-based leg runner: pulls legs from results/analysis/leg_queue.txt
# (ARM|PHRASES|MODE|LAYSET per line), claims each via push-arbitration
# (same pattern as v10 cell claims: only a successful push owns the claim),
# runs roll_assigned_leg.sh per claimed leg, exits when the queue is empty.
cd /workspace/phrase-rl
while :; do
  timeout 180 git -c rebase.autoStash=true pull -q --rebase 2>/dev/null \
    || { git rebase --abort 2>/dev/null; git reset --hard -q origin/main; }
  got=""
  while IFS="|" read -r ARM PHRASES MODE LAYSET; do
    [ -z "$ARM" ] && continue
    [ -f "results/analysis/legclaims/$ARM.claim" ] && continue
    mkdir -p results/analysis/legclaims
    echo "${POD:-x}" > "results/analysis/legclaims/$ARM.claim"
    if timeout 200 bash -c "git add 'results/analysis/legclaims/$ARM.claim' && git commit -q -m 'legclaim $ARM [${POD:-x}]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" 2>/dev/null \
       && grep -q "${POD:-x}" "results/analysis/legclaims/$ARM.claim" 2>/dev/null; then
      got=1
      echo "[legs $(date +%H:%M)] ${POD:-x} claimed $ARM" >> /workspace/leg.log
      POD="${POD:-x}" ARM="$ARM" PHRASES="$PHRASES" MODE="$MODE" LAYSET="$LAYSET" bash scripts/roll_assigned_leg.sh
      break
    fi
    git rebase --abort 2>/dev/null; git reset --hard -q origin/main
  done < results/analysis/leg_queue.txt
  if [ -z "$got" ]; then
    echo "[legs $(date +%H:%M)] queue empty for ${POD:-x} -> cell consumer" >> /workspace/leg.log
    exec env POD="${POD:-x}" ROLL_ONLY=1 REPEATS=1 bash scripts/v10_gen_roll_loop.sh
  fi
done
