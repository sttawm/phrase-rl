#!/usr/bin/env bash
# Diagnostic-study rollout shard: one TASK's phrases, x12 reps, 24 layouts, CRN.
# Usage: run_rollout_shard.sh <task_env_name> [--record]
# Publishes rollouts_study_<task>.parquet, then SELF-STOPS the pod.
set -euxo pipefail
TASK="$1"; RECORD="${2:-}"
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl
git pull --no-edit || true
mkdir -p data

.venv-shim() { :; }
/workspace/INT-ACT/.venv/bin/python - <<PY
import pandas as pd
d = pd.read_parquet("data/study_phrases_rollable.parquet")
d = d[d.task == "$TASK"]
assert len(d) == 19, f"expected 19 phrases for $TASK, got {len(d)}"
d.to_parquet("data/shard_phrases.parquet", index=False)
print(d[["arm","phrase"]].to_string())
PY

REC_ARGS=""
if [ "$RECORD" = "--record" ]; then
  mkdir -p /workspace/study_traj
  REC_ARGS="--record-dir /workspace/study_traj --record-success-only"
fi

cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/shard_phrases.parquet \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --repeats 12 $REC_ARGS \
  --out "/workspace/phrase-rl/data/rollouts_study_${TASK}.parquet"

cd /workspace/phrase-rl
mkdir -p results/overnight/raw
cp -f "data/rollouts_study_${TASK}.parquet" results/overnight/raw/
git add results/overnight/raw/
git -c user.name=shard -c user.email=pod@runpod commit -m "study rollouts: ${TASK} x12 [shard]" || true
git -c user.name=shard -c user.email=pod@runpod pull --rebase --no-edit || true
git push || true
echo "SHARD-DONE ${TASK}"
# self-stop THIS pod (pod-scoped key can only stop itself)
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_POD_ID=')" || true
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_API_KEY=')" || true
runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1 || true
# FOREGROUND stop: nohup+& dies with the tmux server when this is the last session
sleep 20
runpodctl stop pod "$RUNPOD_POD_ID" && echo SELF-STOPPED
