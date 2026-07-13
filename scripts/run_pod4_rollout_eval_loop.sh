#!/usr/bin/env bash
# Pod-4 CONTINUOUS eval loop for the ROLLOUT-REWARD arm (pod 2).
# Watches phase2_rollout/latest (not best_val — the margin-selector proved lossy);
# for each NEW checkpoint step: pull adapter, greedy red-team phrases, x12 CRN
# rollouts on the 24-layout val grid, publish rollouts_rt_rollout_s<step>.parquet.
set -uxo pipefail   # loop survives transient failures
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
P2_SCP="scp -q -r -o StrictHostKeyChecking=no -P 22159 -i $HOME/.ssh/id_ed25519"
cd /workspace/phrase-rl
LAST_STEP=""

while true; do
  git pull --no-edit || true
  rm -rf /tmp/rollout_ckpt
  $P2_SCP root@69.30.85.249:/workspace/phrase-rl/results/checkpoints/phase2_rollout/latest /tmp/rollout_ckpt || { sleep 600; continue; }
  STEP=$(grep -oE '"step": ?[0-9]+' /tmp/rollout_ckpt/trainer_state.json 2>/dev/null | grep -oE '[0-9]+' | head -1)
  [ -z "$STEP" ] && { sleep 600; continue; }
  ARM="rollout_s${STEP}"
  if [ "$STEP" = "$LAST_STEP" ] || [ -f "results/overnight/raw/rollouts_rt_${ARM}.parquet" ] || [ "$STEP" -lt 24 ]; then
    echo "step $STEP already evaluated or too early; sleeping"
    sleep 900
    continue
  fi
  echo "=== ROLLOUT-EVAL step $STEP ==="
  rm -rf results/checkpoints/eval_adapters/rollout
  mkdir -p results/checkpoints/eval_adapters
  mv /tmp/rollout_ckpt results/checkpoints/eval_adapters/rollout
  bash scripts/eval_ckpt_redteam.sh results/checkpoints/eval_adapters/rollout "$ARM" "data/rollouts_rt_${ARM}.parquet" 12 \
    && LAST_STEP="$STEP" \
    || { echo "eval failed for step $STEP; retrying next cycle"; sleep 600; }
done
