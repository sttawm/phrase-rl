#!/usr/bin/env bash
# Publish a bank-scoring shard while it is still being scored.
#
#   OUT=results/analysis/bank_scores_train_0of2.parquet bash scripts/bank_score_pusher.sh
#
# run_bank_scoring.sh only commits when the whole job ends, so a multi-hour shard
# lives solely on the pod until then -- and unpushed pod work is how this project
# has lost results before. Runs in its own session so it never touches the script
# bash is mid-way through executing.
set -uo pipefail
cd /workspace/phrase-rl
OUT="${OUT:?set OUT to the shard parquet path}"
EVERY="${EVERY:-1200}"
log() { echo "[pusher $(date -u +%H:%M)] $*"; }
log "watching $OUT every ${EVERY}s"

last=""
while true; do
  sleep "$EVERY"
  [ -f "$OUT" ] || continue
  # the scorer rewrites this file every chunk; committing a half-written parquet
  # would publish corruption, so only proceed if it reads back cleanly
  rows=$(/workspace/INT-ACT/.venv/bin/python -c "
import pandas as pd, sys
try:
    d = pd.read_parquet('$OUT')
    print(len(d) if d.z.notna().any() else 0)
except Exception:
    print(-1)
" 2>/dev/null)
  case "$rows" in ''|-1|0) log "unreadable or empty right now; skipping"; continue;; esac
  [ "$rows" = "$last" ] && continue
  git add -f "$OUT" 2>/dev/null
  git diff --cached --quiet -- "$OUT" && continue
  git commit -q -m "bank scores in progress: $(basename "$OUT") at $rows rows" || continue
  ok=0
  for i in 1 2 3; do
    timeout 600 bash -c "git -c rebase.autoStash=true pull -q --rebase origin main && git push -q origin HEAD:main" \
      && { ok=1; break; }
    git rebase --abort 2>/dev/null; sleep 20
  done
  [ "$ok" = 1 ] && { log "pushed $rows rows"; last="$rows"; } || log "PUSH FAILED, will retry next round"
  # the scorer exits and pushes on its own; stop once it is gone and we are current
  pgrep -f "[s]core_bank.py" >/dev/null || { log "scorer finished; final state pushed"; exit 0; }
done
