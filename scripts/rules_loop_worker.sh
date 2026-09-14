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
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|VLA_DATA_DIR|VLA_LOG_DIR|WANDB_MODE)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
RUN_ID="${RUN_ID:?set RUN_ID}"
JOBS="results/rules_runs/$RUN_ID/jobs"
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_POD_ID=')" 2>/dev/null || true
POD="${RUNPOD_POD_ID:-$(hostname)}"
# identity must live in the repo config, not the launching session's env: a
# worker relaunched in a fresh tmux lost it and wedged on an un-committable result
git config user.email >/dev/null 2>&1 || git config user.email "worker@phrase-rl.local"
git config user.name  >/dev/null 2>&1 || git config user.name "phrase-rl worker $POD"
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
# salvage: a worker killed mid-job leaves its finished child's result parquet
# uncommitted in the tree. Commit any settled (>60s old) orphan on startup so
# the claimed job completes instead of sitting until the 6h stale-steal.
for orph in "$JOBS"/*.result.parquet; do
  [ -e "$orph" ] || continue
  git ls-files --error-unmatch "$orph" >/dev/null 2>&1 && continue
  [ "$(( $(date +%s) - $(stat -c %Y "$orph") ))" -gt 60 ] || continue
  git add "$orph" && mark "salvaged orphan $(basename "$orph")"
done
git commit -q -m "rules-loop orphan result salvage ($POD)" 2>/dev/null \
  && timeout 300 bash -c "git -c rebase.autoStash=true pull -q --rebase && git push -q"
while true; do
  # graceful drain: touch /workspace/.worker_stop and the worker exits between
  # jobs (rolling restarts without orphaning a claim)
  [ -f /workspace/.worker_stop ] && { mark "stop file -- worker exiting"; exit 0; }
  # in-loop orphan salvage: a result written by a prior worker's child AFTER
  # this worker's startup pass (drain race, 2026-09-02) would otherwise sit
  # uncommitted forever -- the startup-only pass skips files <60s old
  for orph in "$JOBS"/*.result.parquet; do
    [ -e "$orph" ] || continue
    git ls-files --error-unmatch "$orph" >/dev/null 2>&1 && continue
    [ "$(( $(date +%s) - $(stat -c %Y "$orph") ))" -gt 60 ] || continue
    git add "$orph" && mark "salvaged orphan $(basename "$orph")"
    git commit -q -m "rules-loop orphan result salvage ($POD)" 2>/dev/null \
      && timeout 300 bash -c "git -c rebase.autoStash=true pull -q --rebase && git push -q"
  done
  timeout 240 git -c rebase.autoStash=true pull -q --rebase 2>/dev/null \
    || { git rebase --abort 2>/dev/null; git reset --hard -q origin/main; }
  did=0
  for specf in "$JOBS"/*.spec.json; do
    [ -e "$specf" ] || continue
    jid=$(basename "$specf" .spec.json)
    [ -f "$JOBS/$jid.result.parquet" ] && continue

    kind=$(grep -o '"kind": *"[a-z]*"' "$specf" | grep -o '[a-z]*"$' | tr -d '"')
    # apply AND generate jobs need the LLM stack (torch/transformers); only pods
    # marked apply-capable take them. This check must precede the claim: claiming a
    # job this pod then skips wedges it for the 6h stale window (rv2, 2026-09-02).
    # generate was added later (234fd8e6) and was NOT covered here: a LIBERO
    # rollout pod (pe5) claimed the qwen naturals job and burned all 3 attempts on
    # ModuleNotFoundError: torch, FINAL-failing it (a00qwnatgen, 2026-09-04).
    case "$kind" in
      apply|generate) [ -f /workspace/.can_apply ] || continue ;;
    esac

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

    if [ "$kind" = "apply" ]; then
      # Qwen apply needs the whole GPU: 9B bf16 cannot fit beside the resident
      # score server on 24GB. Stop it; the next score job reboots it.
      tmux kill-session -t rlscore 2>/dev/null || true
      pkill -f '[p]hase2_score_server' 2>/dev/null || true
      sleep 8
    fi
    if [ "$kind" = "score" ] && ! grep -qE '"method": "(rollout|libero_bank_eval)"' "$specf"; then
      ensure_score_server
    fi
    mark "running $jid ($kind)"
    att=$(ls "$JOBS/$jid".attempt-* 2>/dev/null | wc -l | tr -d " ")
    if [ "$att" -ge 3 ]; then mark "giving up on $jid after $att attempts"; continue; fi
    touch "$JOBS/$jid.attempt-$((att + 1))"
    if IPC_DIR="$IPC_DIR" .venv-gen/bin/python scripts/rules_loop_jobs.py "$specf" \
         > "/workspace/rljob_$jid.log" 2>&1; then
      git add "$JOBS/$jid.result.parquet"
      [ -f "$JOBS/$jid.episodes.parquet" ] && git add "$JOBS/$jid.episodes.parquet"
      # stage the marker deletion too: an unstaged rm leaves a stale committed
      # failed.txt on origin, which killed a driver mid-wait (2026-08-31)
      git rm -q -f --ignore-unmatch "$JOBS/$jid.failed.txt"
    else
      # retryable marker: only the 3rd failure is FINAL (drivers ignore the rest)
      { [ "$((att + 1))" -ge 3 ] && echo "FINAL attempt $((att + 1))/3"; } \
        > "$JOBS/$jid.failed.txt" || true
      tail -c 2000 "/workspace/rljob_$jid.log" >> "$JOBS/$jid.failed.txt"
      git add "$JOBS/$jid.failed.txt" "$JOBS/$jid.attempt-$((att + 1))" 2>/dev/null \
        || git add "$JOBS/$jid.failed.txt"
      mark "FAILED $jid (attempt $((att + 1))/3)"
    fi
    if ! git commit -q -m "rules-loop result $jid"; then
      # a silent commit failure wedged a finished 4h job on 2026-08-31 (lost git
      # identity): fail LOUDLY and leave the loop so the stall is visible
      mark "COMMIT FAILED for $jid -- worker halting (fix git state, restart)"
      exit 1
    fi
    pushed=0
    for i in 1 2 3; do
      timeout 300 bash -c "git -c rebase.autoStash=true pull -q --rebase && git push -q" \
        && { pushed=1; break; }
      git rebase --abort 2>/dev/null; sleep 20
    done
    [ "$pushed" = 1 ] || mark "PUSH FAILED 3x for $jid -- result committed locally, will retry on next pass"
    did=1
  done
  [ "$did" = 0 ] && sleep 60
done
