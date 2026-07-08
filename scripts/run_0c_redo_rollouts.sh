#!/usr/bin/env bash
# Deferred 0c-redo rollout grid on the CoVer-template phrase sets. Credit-free.
set -euxo pipefail
export UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR="${VLA_DATA_DIR:-/workspace/vla_data}" VLA_LOG_DIR="${VLA_LOG_DIR:-/workspace/vla_log}" WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl
[ -d /workspace/INT-ACT ] || bash scripts/setup_simpler.sh
cd /workspace/INT-ACT
.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/phrases_0c_redo.parquet \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 \
  --out /workspace/phrase-rl/data/rollouts_0c_redo.parquet
cd /workspace/phrase-rl
mkdir -p results/overnight/raw && cp -f data/rollouts_0c_redo.parquet results/overnight/raw/
git add results/overnight && git -c user.name=pod -c user.email=pod@runpod commit -m "0c-redo rollouts [pod]" || true
git -c user.name=pod -c user.email=pod@runpod pull --rebase --no-edit || true; git push || true
echo "0C REDO ROLLOUTS DONE"
