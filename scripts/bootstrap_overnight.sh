#!/usr/bin/env bash
# Fresh-pod bootstrap -> overnight chain. Handles the cu128-wheels-on-older-driver
# case by falling back to cu124 torch in .venv-gen (matches .venv's known-good).
set -euxo pipefail
export UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy
cd /workspace/phrase-rl
bash scripts/setup_pod.sh
bash scripts/setup_gen_env.sh
if ! .venv-gen/bin/python -c "import torch; assert torch.cuda.is_available()"; then
  ~/.local/bin/uv pip install --python .venv-gen/bin/python --reinstall \
    "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124
  .venv-gen/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA after cu124 fallback'"
fi
bash scripts/run_overnight.sh
