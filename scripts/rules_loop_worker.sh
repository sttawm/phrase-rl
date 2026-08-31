#!/bin/bash
# Pod-side worker for the rules loop: polls the run's jobs/ directory on origin,
# executes score/apply jobs, commits results back. Multiple workers may share a
# run: each job is claimed by committing <jid>.claim and pushing -- the push is
# the atomic arbiter (a lost race conflicts on rebase and the loser skips). A
# claim older than 6h with no result is treated as dead and may be stolen.
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
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_POD_ID=')" 2>/dev/null || true
POD="${RUNPOD_POD_ID:-$(hostname)}"
mark() { echo "[rlworker $(date -u +%H:%M)] $*" >> /workspace/rules_worker.log; }

IPC_DIR="${IPC_DIR:-/workspace/ipc_rules}"
case "$IPC_DIR" in
  ""|"/"|"/workspace"|"/root"|"/tmp"|*..*) echo "unsafe IPC_DIR=$IPC_DIR" >&2; exit 1;;
esac

ensure_score_server() {
  # liveness by PROCESS, not by tmux session: a dead server inside a live session
  # is the failure mode that looks healthiest
  if pgrep -f "[p]hase2_score_server.*$IPC_DIR" >/dev/null 2>&1; then return 0; fi
  tmux kill-session -t rlscore 2>/dev/null
  mkdir -p "$IPC_DIR"
  # verifier mode needs the ensemble + stats contexts, or every grip comes back NaN
  # secrets INSIDE the tmux command: a new tmux session inherits the tmux
  # server's original env, not this shell's exports -- the score server booted
  # without HF_TOKEN and died on the gated PaliGemma repo (2026-08-31)
  tmux new-session -d -s rlscore \
    "cd /workspace/phrase-rl && eval \"\$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)\" && export HF_HOME=\${HF_HOME:-/workspace/hf_cache} && .venv/bin/python -m phrase_rl.phase2_score_server \
       --ipc-dir \"$IPC_DIR\" \
       --stats-contexts results/phrase_artifacts/chunk_stats.parquet \
       --verifier-ensemble results/checkpoints/verifier_reward_ensemble_4f.json \
       > /workspace/rl_score_server.log 2>&1"
  mark "booted score server (verifier, ipc=$IPC_DIR)"
  for _ in $(seq 1 40); do
    sleep 15
    pgrep -f "[p]hase2_score_server.*$IPC_DIR" >/dev/null 2>&1 && return 0
  done
  mark "score server FAILED to start"; return 1
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

    # --- claim arbitration (multi-pod) ------------------------------------
    clm="$JOBS/$jid.claim"
    if [ -f "$clm" ]; then
      owner=$(awk '{print $1}' "$clm")
      age=$(( $(date +%s) - $(awk '{print $2}' "$clm" 2>/dev/null || echo 0) ))
      if [ "$owner" != "$POD" ] && [ "$age" -lt 21600 ]; then continue; fi
    fi
    if [ ! -f "$clm" ] || [ "$(awk '{print $1}' "$clm")" != "$POD" ]; then
      echo "$POD $(date +%s)" > "$clm"
      git add "$clm"
      git commit -q -m "rules-loop claim $jid ($POD)"
      # pull --rebase: if another pod's claim landed first, replaying ours
      # conflicts on the same file -- that conflict IS losing the race
      if ! timeout 180 bash -c "git -c rebase.autoStash=true pull -q --rebase && git push -q"; then
        git rebase --abort 2>/dev/null
        git reset --hard -q origin/main
        mark "lost claim race on $jid"
        continue
      fi
      mark "claimed $jid"
    fi
    # ----------------------------------------------------------------------

    kind=$(grep -o '"kind": *"[a-z]*"' "$specf" | grep -o '[a-z]*"$' | tr -d '"')
    if [ "$kind" = "apply" ]; then
      # Qwen apply needs the whole GPU: 9B bf16 cannot fit beside the resident
      # score server on 24GB. Stop it; the next score job reboots it.
      tmux kill-session -t rlscore 2>/dev/null || true
      pkill -f '[p]hase2_score_server' 2>/dev/null || true
      sleep 8
    fi
    [ "$kind" = "score" ] && ensure_score_server
    mark "running $jid ($kind)"
    att=$(ls "$JOBS/$jid".attempt-* 2>/dev/null | wc -l | tr -d " ")
    if [ "$att" -ge 3 ]; then mark "giving up on $jid after $att attempts"; continue; fi
    touch "$JOBS/$jid.attempt-$((att + 1))"
    if IPC_DIR="$IPC_DIR" .venv-gen/bin/python scripts/rules_loop_jobs.py "$specf" \
         > "/workspace/rljob_$jid.log" 2>&1; then
      git add "$JOBS/$jid.result.parquet"
      rm -f "$JOBS/$jid.failed.txt"
    else
      # retryable marker: a transient scoring timeout must not poison the job id
      tail -c 2000 "/workspace/rljob_$jid.log" > "$JOBS/$jid.failed.txt"
      git add "$JOBS/$jid.failed.txt" "$JOBS/$jid.attempt-$((att + 1))"
      mark "FAILED $jid (attempt $((att + 1))/3)"
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
