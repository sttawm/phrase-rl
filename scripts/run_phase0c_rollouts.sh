#!/usr/bin/env bash
# Full Phase 0c rollout grid on the pod. Resumable (re-run to continue).
set -euxo pipefail
export UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|VLA_DATA_DIR|VLA_LOG_DIR|WANDB_MODE)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR="${VLA_DATA_DIR:-/workspace/vla_data}" VLA_LOG_DIR="${VLA_LOG_DIR:-/workspace/vla_log}"
export WANDB_MODE=offline

cd /workspace/INT-ACT
.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/phrases_0c.parquet \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 \
  --out /workspace/phrase-rl/data/rollouts_0c.parquet

echo "PHASE0C ROLLOUTS DONE"
