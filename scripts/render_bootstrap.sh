#!/bin/bash
# Render-pod bootstrap for --phase sim rollouts (SIMPLER + INT-ACT + Vulkan).
# Run on a fresh RunPod 4090 pod:  bash render_bootstrap.sh
# Expects ~/.bashrc to already carry HF_TOKEN, and /root/.podtok (git PAT).
# Emits RENDER-BOOTSTRAP-DONE on success; VULKAN-UNFIXABLE if the host lacks
# graphics capability (recreate the pod -- do not fight it, see memory).
set -uxo pipefail
exec > /workspace/render_bringup.log 2>&1
eval "$(grep -E '^export HF_TOKEN' ~/.bashrc || true)"
export HF_HOME=/workspace/hf_cache

# --- 0. graphics-capability preflight (unfixable class if absent) -----------
caps=$(tr '\0' '\n' < /proc/1/environ | grep NVIDIA_DRIVER_CAPABILITIES || true)
echo "caps: $caps"
case "$caps" in
  *graphics*|*all*) echo "caps-ok";;
  *) echo "VULKAN-UNFIXABLE (no graphics capability)"; exit 7;;
esac

# --- 1. system deps + vulkan loader stack ------------------------------------
apt-get update -qq
apt-get install -y -qq git-lfs ffmpeg tmux libgl1 libegl1 libgles2 \
  libvulkan1 vulkan-tools libglvnd0 libglvnd-dev
# loader only reads /usr/share/vulkan/icd.d (memory: /etc copy is ignored)
mkdir -p /usr/share/vulkan/icd.d
if [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  src=$(ls /etc/vulkan/icd.d/nvidia_icd.json 2>/dev/null | head -1)
  if [ -n "$src" ]; then cp "$src" /usr/share/vulkan/icd.d/; else
    cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}
JSON
  fi
fi
vulkaninfo --summary 2>&1 | head -5 || true
if vulkaninfo --summary 2>&1 | grep -qi llvmpipe && ! vulkaninfo --summary 2>&1 | grep -qi nvidia; then
  echo "VULKAN-LLVMPIPE-ONLY"; exit 7
fi

# --- 2. uv --------------------------------------------------------------------
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH=$HOME/.local/bin:$PATH

# --- 3. phrase-rl clone (blob-filtered, sparse) + worker venv ----------------
TOK=$(cat /root/.podtok)
printf 'https://x-access-token:%s@github.com\n' "$TOK" > /root/.git-credentials-phrase-rl
chmod 600 /root/.git-credentials-phrase-rl
rm -f /root/.podtok
git config --global credential.helper "store --file=/root/.git-credentials-phrase-rl"
git config --global user.email "worker@phrase-rl.local"
git config --global user.name  "phrase-rl render worker"
if [ ! -d /workspace/phrase-rl/.git ]; then
  git clone --filter=blob:none --no-checkout https://github.com/sttawm/phrase-rl.git /workspace/phrase-rl
  cd /workspace/phrase-rl
  git sparse-checkout init --no-cone
  printf '/*\n!/results/checkpoints/archive/*\n!/results/analysis/data_archive/*\n' > .git/info/sparse-checkout
  git checkout -q main
fi
cd /workspace/phrase-rl
# jobs.py runtime only (pandas/pyarrow/numpy): rollout math runs in INT-ACT's venv
if [ ! -x .venv-gen/bin/python ]; then
  uv venv --python 3.11 .venv-gen
  uv pip install -p .venv-gen/bin/python pandas pyarrow numpy
fi
echo CLONE-DONE

# --- 4. INT-ACT + its venv ----------------------------------------------------
if [ ! -d /workspace/INT-ACT ]; then
  git clone https://github.com/ai4ce/INT-ACT.git /workspace/INT-ACT
fi
cd /workspace/INT-ACT
git submodule update --init --recursive   # third_party/lerobot etc. are submodules
if ! .venv/bin/python -c "import simpler_env" 2>/dev/null; then
  rm -rf .venv
  uv venv --python 3.10.12 .venv        # INT-ACT pins ==3.10.12 exactly
  uv sync --python .venv/bin/python
  uv pip install -p .venv/bin/python 'setuptools<81'   # pkg_resources (removed in 81+)
  .venv/bin/python -c "import simpler_env" 2>/dev/null ||     uv pip install -p .venv/bin/python -e third_party/SimplerEnv -e third_party/ManiSkill2_real2sim
fi
.venv/bin/python -c "import simpler_env; print('simpler-ok')" \
  || { echo "SIMPLER-IMPORT-FAILED"; exit 8; }

# --- 5. models ---------------------------------------------------------------
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("juexzz/INTACT-pi0-finetune-rephrase-bridge")
print("ckpt-ok")
PY

echo RENDER-BOOTSTRAP-DONE
