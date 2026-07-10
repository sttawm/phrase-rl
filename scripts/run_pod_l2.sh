#!/usr/bin/env bash
# Phase 2 — L2 reward arm (Arm B). Identical to run_phase2.sh EXCEPT the reward:
# the score server computes CRN decoded-action per-DoF-normalized L2 vs the
# ground-truth chunk (reward_mode=l2) instead of the flow-matching residual.
# Runs on its OWN pod, its OWN checkpoint tree (phase2_l2), so it trains in
# parallel with the flow arm (run_phase2.sh) without collision.
#
# Fair A/B: set INIT_ADAPTER to the SAME checkpoint the flow arm started from
# (v1 step-200 best_val) so both arms diverge from an identical rephraser under
# different rewards. Ship that adapter to the pod first, then:
#   INIT_ADAPTER=results/checkpoints/phase2_v1_init/best_val bash scripts/run_pod_l2.sh
# (omit INIT_ADAPTER to start from base Qwen.)
set -euxo pipefail
export UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl

if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "GEMINI_API_KEY not set (gemini judge needs it)"; exit 1
fi

STATS_CONTEXTS="${STATS_CONTEXTS:-data/contexts_train.parquet}"
if [ ! -f "$STATS_CONTEXTS" ]; then
  echo "missing $STATS_CONTEXTS — the L2 reward needs it for per-DoF normalization."
  echo "copy it: cp results/phrase_artifacts/contexts_train.parquet data/  (or wherever it lives)"
  exit 1
fi

IPC_DIR=/workspace/ipc_l2
CKPT_DIR=results/checkpoints/phase2_l2
mkdir -p "$IPC_DIR" "$CKPT_DIR"
rm -f "$IPC_DIR"/*.req.json "$IPC_DIR"/*.done.json "$IPC_DIR"/*.failed.json \
      "$IPC_DIR"/*.err.txt "$IPC_DIR"/*.parquet "$IPC_DIR"/*.tmp

NGPU=$(nvidia-smi -L | wc -l)
SCORE_GPU=0
TRAIN_GPU=$(( NGPU >= 2 ? 1 : 0 ))

tmux kill-session -t score 2>/dev/null || true
tmux kill-session -t train 2>/dev/null || true

# NEVER `uv run` here — it re-syncs and uninstalls packages. Direct venv pythons.
tmux new-session -d -s score \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY|UV_CACHE_DIR|UV_LINK_MODE)\" ~/.bashrc || true)\"; cd /workspace/phrase-rl && \
   CUDA_VISIBLE_DEVICES=$SCORE_GPU .venv/bin/python -m phrase_rl.phase2_score_server \
     --ipc-dir $IPC_DIR --stats-contexts $STATS_CONTEXTS \
   2>&1 | tee -a $CKPT_DIR/score_server.log; \
   echo \"score server exited rc=\$?\"; sleep infinity'"

tmux new-session -d -s train \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY|UV_CACHE_DIR|UV_LINK_MODE)\" ~/.bashrc || true)\"; cd /workspace/phrase-rl && \
   CUDA_VISIBLE_DEVICES=$TRAIN_GPU .venv-gen/bin/python -m phrase_rl.phase2_train \
     --ipc-dir $IPC_DIR --ckpt-dir $CKPT_DIR --resume \
     --reward-mode l2 --k-l2 4 \
     --beta 0.15 --lr 1e-5 --kl-abort 1.2 --contexts-per-step 6 \
     ${JUDGE_BACKEND:+--judge-backend $JUDGE_BACKEND --judge-model gemini-3.5-flash} \
     ${INIT_ADAPTER:+--init-adapter $INIT_ADAPTER} \
     --traces results/phrase_artifacts/cover35_teacher_train.parquet \
     --probe-contexts results/phrase_artifacts/contexts_0c_tasks.parquet \
     --probe-traces results/phrase_artifacts/traces_0c_tasks.parquet --probe-every 25 \
     --val-traces results/phrase_artifacts/rephrases_val_0b.parquet  # INTERIM val traces — keep constant for the whole run \
   2>&1 | tee -a $CKPT_DIR/train.log; \
   echo \"trainer exited rc=\$? (0=clean/interrupted 3=gate unavailable 4=score timeout 5=score server error 6=KL abort)\"; \
   sleep infinity'"

set +x
cat <<EOF

Phase 2 L2 ARM launched (score on GPU $SCORE_GPU, train on GPU $TRAIN_GPU).
  reward       : CRN decoded-action per-DoF-normalized L2 vs a* (reward_mode=l2, k_l2=4)
  score server : tmux attach -t score    log: $CKPT_DIR/score_server.log
  trainer      : tmux attach -t train    log: $CKPT_DIR/train.log
  step metrics : tail -f $CKPT_DIR/train_log.jsonl
  checkpoints  : $CKPT_DIR/{latest,best_val,final}

NOTE: each L2 reward decode is a FULL denoise (k_l2=4 per phrase), so steps are
slower than the flow arm — judge by early vals (@100, @200), same as flow.
Re-run this script to restart both; the trainer resumes from $CKPT_DIR/latest.
EOF
