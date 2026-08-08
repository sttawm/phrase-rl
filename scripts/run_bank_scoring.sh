#!/bin/bash
# Pod-side launcher for phrase-bank scoring (rules-loop initialisation).
#
#   TASK_KIND=train SHARD=0 OF=2 bash scripts/run_bank_scoring.sh
#
# TASK_KIND=train scores the 213 bridge training instructions (needs the club
# context table); TASK_KIND=sim scores the 8 widowx simulator tasks the loop
# validates on (needs only the val contexts, so it runs on any pod).
#
# Waits out any in-flight git gc, boots a verifier score server if none is up,
# then scores. Results are flushed per task and pushed, so an interruption costs
# one task rather than the night.
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
TASK_KIND="${TASK_KIND:?set TASK_KIND to train or sim}"
SHARD="${SHARD:-0}"; OF="${OF:-1}"
IPC_DIR="${IPC_DIR:-/workspace/ipc_bank}"
case "$IPC_DIR" in
  ""|"/"|"/workspace"|"/root"|"/tmp"|*..*) echo "unsafe IPC_DIR=$IPC_DIR" >&2; exit 1;;
esac
LOG=/workspace/bank_${TASK_KIND}_${SHARD}.log
mark() { echo "[bank $(date -u +%H:%M)] $*" | tee -a "$LOG"; }

# Pods disagree on which venv exists: e1/e6 have no .venv at all, only .venv-gen
# and the INT-ACT one. Hardcoding an interpreter fails hours into a job, so pick
# one that can actually import the package. INT-ACT's venv is preferred -- it is
# the runtime already proven to drive pi0.
pick_python() {
  local want="$1"
  for v in /workspace/INT-ACT/.venv .venv-gen .venv; do
    [ -x "$v/bin/python" ] || continue
    if PYTHONPATH=/workspace/phrase-rl/src "$v/bin/python" -c "import $want" >/dev/null 2>&1; then
      echo "$v/bin/python"; return 0
    fi
  done
  return 1
}


# a repack in flight holds the index lock; pulling under it corrupts nothing but
# fails loudly and repeatedly, so just wait it out
for _ in $(seq 1 240); do pgrep -f "[g]it gc|[g]it repack" >/dev/null || break; sleep 30; done
timeout 300 git -c rebase.autoStash=true pull -q --rebase origin main 2>/dev/null \
  || { git rebase --abort 2>/dev/null; git fetch -q origin main && git reset --hard -q origin/main; }
mark "at $(git rev-parse --short HEAD)"

if [ "$TASK_KIND" = train ] && [ ! -f data/contexts_club.parquet ]; then
  mark "FATAL: train scoring needs data/contexts_club.parquet (rebuild: scripts/rebuild_club_contexts.sh)"
  exit 1
fi

SRV_PY=$(pick_python phrase_rl.phase2_score_server) \
  || { mark "FATAL: no venv on this pod can import phrase_rl.phase2_score_server"; exit 1; }
SCORE_PY=$(pick_python phrase_rl.phase2_train) \
  || { mark "FATAL: no venv can import phrase_rl.phase2_train (needs google.genai)"; exit 1; }
mark "server python: $SRV_PY"

if ! pgrep -f "[p]hase2_score_server.*$IPC_DIR" >/dev/null 2>&1; then
  tmux kill-session -t bankscore 2>/dev/null
  mkdir -p "$IPC_DIR"
  tmux new-session -d -s bankscore \
    "cd /workspace/phrase-rl && PYTHONPATH=/workspace/phrase-rl/src $SRV_PY -m phrase_rl.phase2_score_server \
       --ipc-dir \"$IPC_DIR\" \
       --stats-contexts results/phrase_artifacts/chunk_stats.parquet \
       --verifier-ensemble results/checkpoints/verifier_reward_ensemble_4f.json \
       > /workspace/bank_score_server.log 2>&1"
  mark "booting verifier score server (ipc=$IPC_DIR)"
  ok=0
  for _ in $(seq 1 40); do
    sleep 15
    pgrep -f "[p]hase2_score_server.*$IPC_DIR" >/dev/null 2>&1 && { ok=1; break; }
  done
  [ "$ok" = 1 ] || { mark "FATAL score server did not start"; tail -30 /workspace/bank_score_server.log | tee -a "$LOG"; exit 1; }
fi
mark "score server up; scoring $TASK_KIND shard $SHARD/$OF"

IPC_DIR="$IPC_DIR" PYTHONPATH=/workspace/phrase-rl/src $SCORE_PY scripts/score_bank.py \
  --task-kind "$TASK_KIND" --shard "$SHARD" --of "$OF" --ipc "$IPC_DIR" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

OUT="results/analysis/bank_scores_${TASK_KIND}_${SHARD}of${OF}.parquet"
if [ -f "$OUT" ]; then
  git add -f "$OUT"
  git commit -q -m "bank scores: $TASK_KIND shard $SHARD/$OF (rc=$rc)" || true
  for i in 1 2 3; do
    timeout 600 bash -c "git -c rebase.autoStash=true pull -q --rebase origin main && git push -q origin HEAD:main" \
      && { mark "pushed $OUT"; break; }
    git rebase --abort 2>/dev/null; sleep 20
  done
fi
mark "BANK-SCORING-EXIT rc=$rc kind=$TASK_KIND shard=$SHARD"
