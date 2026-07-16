#!/usr/bin/env bash
# One-shot L40S training-pod bootstrap + arm migration + 22h auto-stop.
# Usage (on the NEW pod, after repo clone): bash scripts/bootstrap_trainpod.sh A|B
# Expects: ~/.bashrc secrets appended, checkpoint dir + data shipped separately
# (the Mac-side migrate script handles both).
set -euxo pipefail
ARM="${1:?usage: bootstrap_trainpod.sh A|B}"
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl

# container-disk venvs (MFS-safe), pytest+av runtime deps
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=hardlink
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH=$HOME/.local/bin:$PATH
apt-get update -qq && apt-get install -y -qq git-lfs tmux ffmpeg libgl1 && git lfs install
if [ ! -x /root/venv-pi0/bin/python ]; then
  uv venv --python 3.11 /root/venv-pi0 && ln -sfn /root/venv-pi0 .venv
  uv pip install -p /root/venv-pi0/bin/python -e ".[gpu]"
  uv pip install -p /root/venv-pi0/bin/python \
    "lerobot @ git+https://github.com/IrvingF7/lerobot.git@35f6e02315dcb3fa25f3a740478265c6c793a95e" \
    "transformers==4.48.3" pytest av
fi
if [ ! -x /root/venv-gen/bin/python ]; then
  rm -rf /root/.cache/uv
  uv venv --python 3.11 /root/venv-gen && ln -sfn /root/venv-gen .venv-gen
  uv pip install -p /root/venv-gen/bin/python "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128
  uv pip install -p /root/venv-gen/bin/python transformers accelerate pillow pandas pyarrow peft google-genai tqdm
  uv pip install -p /root/venv-gen/bin/python -e . --no-deps
  rm -rf /root/.cache/uv
fi
/root/venv-pi0/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('juexzz/INTACT-pi0-finetune-rephrase-bridge')"
/root/venv-gen/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-9B')"

# allocator experiment: expandable segments (backward-stall hypothesis, 2026-07-16)
grep -q PYTORCH_CUDA_ALLOC_CONF ~/.bashrc || echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc

# 22h auto-stop guard (foreground-safe: setsid survives this script's exit;
# uses its own session so tmux teardown can't kill it)
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_POD_ID=')" || true
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_API_KEY=')" || true
runpodctl config --apiKey "$RUNPOD_API_KEY" 2>&1 | tail -1 || true
setsid bash -c "sleep $((22*3600)); runpodctl stop pod $RUNPOD_POD_ID >> /workspace/autostop.log 2>&1" < /dev/null > /dev/null 2>&1 &

# launch (resumes from shipped checkpoint; parity + banner checks run at startup)
test -f data/contexts_train.parquet && test -f data/contexts_train_multit.parquet
REWARD_FRAMES=4 bash scripts/run_arm.sh "$ARM"
echo "BOOTSTRAP-DONE-$ARM"
