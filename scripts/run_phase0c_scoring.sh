#!/usr/bin/env bash
# Waits for the 0c rollout grid to finish, then CRN-scores the task phrases on
# their matched real Bridge contexts. Resumable.
set -euxo pipefail
export UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl

while tmux has-session -t rollout 2>/dev/null; do sleep 60; done
echo "rollouts done; starting 0c flow-loss scoring"

source .venv/bin/activate
python -m phrase_rl.phase0c_score \
  --contexts data/contexts_0c_match.parquet \
  --phrases data/phrases_0c.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --out data/scores_0c.parquet
echo "PHASE0C SCORING DONE"
