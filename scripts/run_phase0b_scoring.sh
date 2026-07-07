#!/usr/bin/env bash
# Waits for the qwengen tmux session (candidate generation) to finish, then
# CRN-scores both INTACT checkpoints over both phrase arms. Resumable.
set -euxo pipefail
cd "$(dirname "$0")/.."
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

while tmux has-session -t qwengen 2>/dev/null; do sleep 60; done
echo "qwengen finished; starting scoring"

source .venv/bin/activate
ARMS=(--arm gemini=data/rephrases_val_0b.parquet --arm qwen=data/qwen_rephrases_val_0b.parquet)

python -m phrase_rl.phase0b_score --contexts data/contexts_val_0b.parquet "${ARMS[@]}" \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --out data/scores_0b_rephrase.parquet

python -m phrase_rl.phase0b_score --contexts data/contexts_val_0b.parquet "${ARMS[@]}" \
  --ckpt juexzz/INTACT-pi0-finetune-bridge --out data/scores_0b_plain.parquet

echo "PHASE0B SCORING DONE"
