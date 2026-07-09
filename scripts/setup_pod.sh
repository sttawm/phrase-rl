#!/usr/bin/env bash
# One-shot RunPod setup. Run from repo root after cloning:
#   git clone https://github.com/sttawm/phrase-rl.git && cd phrase-rl && bash scripts/setup_pod.sh
# Expects env vars: HF_TOKEN (and later GEMINI_API_KEY for phases 0b+/1).
set -euxo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/uv_cache}" UV_LINK_MODE=copy  # default: big /workspace; override for tight quotas

# --- system deps ---
apt-get update -qq && apt-get install -y -qq git-lfs ffmpeg libgl1 tmux
git lfs install

# --- uv + python env ---
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.11
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[gpu]"

# Reward env: EXACTLY the INTACT training environment — the lerobot fork+commit
# vendored by https://github.com/ai4ce/INT-ACT (old lerobot.common.* layout,
# policy-internal normalization/tokenization, forward(batch, noise, time)).
# transformers is pinned to match; the Qwen phrase model gets its own venv.
uv pip install "lerobot @ git+https://github.com/IrvingF7/lerobot.git@35f6e02315dcb3fa25f3a740478265c6c793a95e" "transformers==4.48.3" pytest

# --- frozen model downloads (HF cache on the network volume) ---
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in [
    "juexzz/INTACT-pi0-finetune-rephrase-bridge",
    "juexzz/INTACT-pi0-finetune-bridge",
]:
    print("downloading", repo)
    snapshot_download(repo)
PY

echo "Setup complete. Next: python -m phrase_rl.smoke_pi0 (Phase 0a smoke test)."
