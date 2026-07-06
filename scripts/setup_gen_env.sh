#!/usr/bin/env bash
# Generation env (.venv-gen): latest transformers stack for the trainable
# phrase VLM. Separate from .venv, which is pinned to INTACT-era lerobot for
# reward scoring (transformers 4.48 vs >=5 conflict).
set -euxo pipefail
cd "$(dirname "$0")/.."

eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
UV="$HOME/.local/bin/uv"

[ -d .venv-gen ] || "$UV" venv .venv-gen --python 3.11
source .venv-gen/bin/activate
"$UV" pip install torch torchvision transformers accelerate pillow pandas pyarrow peft

python - <<'PY'
from huggingface_hub import snapshot_download
print("downloading Qwen/Qwen3.5-9B ...")
snapshot_download("Qwen/Qwen3.5-9B")
print("done")
PY
echo "gen env ready"
