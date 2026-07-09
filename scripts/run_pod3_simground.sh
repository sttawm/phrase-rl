#!/usr/bin/env bash
# Pod-3: sim-grounded reward arm. Record trajectories (original instruction,
# 40 seeds/task) -> extract successful-episode contexts -> CRN-score all redo
# phrases against SIM contexts -> publish. Gemini-free.
set -euxo pipefail
export UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR="${VLA_DATA_DIR:-/workspace/vla_data}" VLA_LOG_DIR="${VLA_LOG_DIR:-/workspace/vla_log}" WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl
git pull --no-edit || true
mkdir -p data
cp -f results/overnight/raw/phrases_0c_redo.parquet data/

# originals-only phrase list for trajectory recording
.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/phrases_0c_redo.parquet")
df[df.arm == "original"].to_parquet("data/phrases_originals.parquet", index=False)
print(df[df.arm == "original"])
PY

cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/phrases_originals.parquet \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 \
  --out /workspace/phrase-rl/data/rollouts_record.parquet \
  --record-dir /workspace/phrase-rl/data/sim_traj --record-success-only
cd /workspace/phrase-rl

.venv/bin/python -m phrase_rl.sim_contexts_extract --record-dir data/sim_traj \
  --out data/contexts_sim.parquet --per-episode 3

source .venv/bin/activate
python -m phrase_rl.phase0c_score --contexts data/contexts_sim.parquet \
  --phrases data/phrases_0c_redo.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --out data/scores_0c_sim.parquet
deactivate

mkdir -p results/overnight/raw
cp -f data/contexts_sim.parquet data/scores_0c_sim.parquet data/rollouts_record.parquet results/overnight/raw/
git add results/overnight && git -c user.name=pod3 -c user.email=pod@runpod commit -m "sim-grounded arm: sim contexts + sim-grounded flow scores [pod]" || true
git -c user.name=pod3 -c user.email=pod@runpod pull --rebase --no-edit || true; git push || true
echo "POD3 SIMGROUND DONE"
