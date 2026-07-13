#!/usr/bin/env bash
# Evaluate ONE adapter checkpoint on the red-team val protocol.
# Usage: eval_ckpt_redteam.sh <adapter_dir> <arm_name> <out_parquet> [reps]
# Generates greedy phrases (Gemini-trace assets), rolls out on the 24-layout val
# grid with CRN, publishes the parquet. set -e: any failure kills the run BEFORE
# the DONE marker (learned from the silent l2late chain failure, 2026-07-12).
set -euxo pipefail
ADAPTER="$1"; ARM="$2"; OUT="$3"; REPS="${4:-12}"
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl

PHRASES="data/phrases_${ARM}.parquet"
.venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
  --tasks results/phrase_artifacts/cover_gemini_tasks.parquet \
  --assets results/phrase_artifacts/redteam_eval_assets.parquet \
  --adapter "$ADAPTER" \
  --out "data/gen_${ARM}.parquet"
ARM="$ARM" PHRASES="$PHRASES" .venv-gen/bin/python - <<'PY'
import os
import pandas as pd
arm, phrases = os.environ["ARM"], os.environ["PHRASES"]
d = pd.read_parquet(f"data/gen_{arm}.parquet")
d = d[d.arm == "tuned"].assign(arm=arm)
d.to_parquet(phrases, index=False)
print(d[["task", "arm", "phrase"]].to_string())
PY

cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases "/workspace/phrase-rl/$PHRASES" \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --repeats "$REPS" \
  --out "/workspace/phrase-rl/$OUT"

cd /workspace/phrase-rl
mkdir -p results/overnight/raw
cp -f "$OUT" "$PHRASES" results/overnight/raw/
git add results/overnight/raw/
git -c user.name=pod3 -c user.email=pod@runpod commit -m "ckpt eval: ${ARM} x${REPS} [pod]" || true
git -c user.name=pod3 -c user.email=pod@runpod pull --rebase --no-edit || true
git push || true
echo "EVAL-CKPT-DONE ${ARM}"
