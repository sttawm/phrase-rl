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
  python3-dev build-essential cmake ninja-build  # evdev needs Python.h; ruckig needs cmake
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

# SimplerEnv + ManiSkill2_real2sim are separate editable installs on top of the
# synced env. ruckig 0.17.3 has stale scikit-build-core metadata -> pin the old
# backend + nanobind and build without isolation. sapien needs pkg_resources ->
# setuptools<70. IMPORTANT: never `uv run` here — it re-syncs to the lockfile
# and silently uninstalls all of these (use .venv/bin/python or --no-sync).
"$UV" pip install "scikit-build-core<0.10" pybind11 nanobind "setuptools<70"
"$UV" pip install ruckig --no-build-isolation
"$UV" pip install -e third_party/ManiSkill2_real2sim -e third_party/SimplerEnv

# --- smoke: create a SIMPLER Bridge env, reset, render one frame ---
.venv/bin/python - <<'PY'
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
