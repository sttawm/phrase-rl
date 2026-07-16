#!/usr/bin/env bash
# Shard-2 (restarted): 4-frame verifier TRAINING features — the native-4f model's
# input. Featurizes contexts_train_multit (2,000 episodes x 4 quartile frames x 3
# conditions = 24,000 rows), publishes, self-stops (foreground pattern).
# Container-disk venv (this volume's MFS eats site-packages writes).
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
git pull --no-edit || true

if [ ! -x /root/venv/bin/python ]; then
  export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=hardlink
  command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
  export PATH=$HOME/.local/bin:$PATH
  uv venv --python 3.11 /root/venv && ln -sfn /root/venv .venv
  uv pip install -p /root/venv/bin/python -e ".[gpu]"
  uv pip install -p /root/venv/bin/python \
    "lerobot @ git+https://github.com/IrvingF7/lerobot.git@35f6e02315dcb3fa25f3a740478265c6c793a95e" \
    "transformers==4.48.3" pytest av
  rm -rf /root/.cache/uv
fi

test -f data/contexts_train_multit.parquet  # shipped from local before launch
/root/venv/bin/python -m phrase_rl.build_verifier_data \
  --contexts data/contexts_train_multit.parquet \
  --stats-contexts results/phrase_artifacts/chunk_stats.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --k 8 --k-decode 4 --out data/verifier_features_train_multit.parquet

mkdir -p results/overnight/raw
cp -f data/verifier_features_train_multit.parquet results/overnight/raw/
git add results/overnight/raw/ \
  && git -c user.name=shard2 -c user.email=pod@runpod commit -m "4-frame verifier training features: 24k rows [pod]" \
  && git -c user.name=shard2 -c user.email=pod@runpod pull --rebase --no-edit \
  && git push
echo MULTIT-FEATURES-DONE

export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_POD_ID=')" || true
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_API_KEY=')" || true
runpodctl config --apiKey "$RUNPOD_API_KEY" 2>&1 | tail -1 || true
sleep 30
runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 | tee -a /workspace/selfstop.log
