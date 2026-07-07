#!/usr/bin/env bash
# Phase 0c infra: INT-ACT (same authors as our checkpoints) bundles SimplerEnv +
# ManiSkill2_real2sim + eval harness for exactly these pi0 checkpoints.
# Headless rendering needs Vulkan + the NVIDIA ICD, which containers often lack.
set -euxo pipefail
export UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy  # container root disk is tiny
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
UV="$HOME/.local/bin/uv"

# --- vulkan headless prerequisites ---
apt-get update -qq
apt-get install -y -qq libvulkan1 vulkan-tools libegl1 libglvnd0 libgl1 libglx0 libxrandr2 \
  python3-dev build-essential  # evdev (via lerobot->pynput) compiles against Python.h
mkdir -p /usr/share/vulkan/icd.d
cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.194"}}
EOF
vulkaninfo --summary 2>&1 | head -20 || echo "WARN: vulkaninfo failed (may still work via EGL)"

# --- INT-ACT + submodules ---
cd /workspace
if [ ! -d INT-ACT ]; then
  git clone --recurse-submodules --shallow-submodules --depth 1 https://github.com/ai4ce/INT-ACT.git
fi
cd INT-ACT
"$UV" sync 2>&1 | tail -5

# --- smoke: create a SIMPLER Bridge env, reset, render one frame ---
"$UV" run python - <<'PY'
import simpler_env
import numpy as np
from PIL import Image
env = simpler_env.make("widowx_spoon_on_towel")
obs, info = env.reset(seed=0)
img = simpler_env.utils.env.observation_utils.get_image_from_maniskill2_obs_dict(env, obs)
Image.fromarray(np.asarray(img)).save("/workspace/simpler_test_frame.png")
print("instruction:", env.get_language_instruction())
print("frame:", np.asarray(img).shape)
print("SIMPLER SMOKE OK")
PY
echo "SIMPLER SETUP DONE"
