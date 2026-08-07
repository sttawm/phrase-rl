#!/bin/bash
# Pod-side worker for the rules loop: polls the run's jobs/ directory on origin,
# executes score/apply jobs, commits results back. One worker per run (v1: no
# claim arbitration -- do not point two workers at the same run).
#
#   RUN_ID=r1 bash scripts/rules_loop_worker.sh
#
# score jobs need the phase2 score server up; this boots one if absent (same
# session pattern as run_arm: tmux 'score', IPC dir /workspace/ipc).
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
RUN_ID="${RUN_ID:?set RUN_ID}"
JOBS="results/rules_runs/$RUN_ID/jobs"
mark() { echo "[rlworker $(date -u +%H:%M)] $*" >> /workspace/rules_worker.log; }

ensure_score_server() {
  tmux has-session -t score 2>/dev/null && return 0
  mkdir -p /workspace/ipc
  tmux new-session -d -s score \
    "cd /workspace/phrase-rl && .venv/bin/python -m phrase_rl.phase2_score_server \
       --ipc-dir /workspace/ipc > /workspace/rl_score_server.log 2>&1"
  mark "booted score server"
  sleep 30
}

mark "worker up for run $RUN_ID"
while true; do
  timeout 240 git -c rebase.autoStash=true pull -q --rebase 2>/dev/null \
    || { git rebase --abort 2>/dev/null; git reset --hard -q origin/main; }
  did=0
  for specf in "$JOBS"/*.spec.json; do
    [ -e "$specf" ] || continue
    jid=$(basename "$specf" .spec.json)
    [ -f "$JOBS/$jid.result.parquet" ] && continue
    [ -f "$JOBS/$jid.failed.txt" ] && continue
    kind=$(grep -o '"kind": *"[a-z]*"' "$specf" | grep -o '[a-z]*"$' | tr -d '"')
    [ "$kind" = "score" ] && ensure_score_server
    mark "running $jid ($kind)"
    if .venv-gen/bin/python scripts/rules_loop_jobs.py "$specf" \
         > "/workspace/rljob_$jid.log" 2>&1; then
      git add "$JOBS/$jid.result.parquet"
    else
      tail -c 2000 "/workspace/rljob_$jid.log" > "$JOBS/$jid.failed.txt"
      git add "$JOBS/$jid.failed.txt"
      mark "FAILED $jid"
    fi
    git commit -q -m "rules-loop result $jid"
    for i in 1 2 3; do
      timeout 300 bash -c "git -c rebase.autoStash=true pull -q --rebase && git push -q" \
        && break
      git rebase --abort 2>/dev/null; sleep 20
    done
    did=1
  done
  [ "$did" = 0 ] && sleep 60
done
