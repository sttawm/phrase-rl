#!/usr/bin/env bash
# Battery rollout shard: rolls a HALF of the baseline battery (by task index
# parity) with N parallel workers, publishes, SELF-STOPS. Phrases parquet must
# already be committed (data/battery_phrases.parquet via git).
# Usage: run_battery_shard.sh 0|1 [n_workers]
set -euxo pipefail
HALF="${1:?usage: run_battery_shard.sh 0|1 [workers]}"
NW="${2:-3}"
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl && git pull --no-edit || true
test -f results/phrase_artifacts/battery_phrases.parquet
cp -f results/phrase_artifacts/battery_phrases.parquet data/battery_phrases.parquet

# split this half's tasks across NW workers
/workspace/INT-ACT/.venv/bin/python - <<PY
import pandas as pd
d = pd.read_parquet("data/battery_phrases.parquet")
tasks = sorted(d.task.unique())
mine = [t for i, t in enumerate(tasks) if i % 2 == $HALF]
print("half $HALF tasks:", mine)
for w in range($NW):
    wt = [t for i, t in enumerate(mine) if i % $NW == w]
    d[d.task.isin(wt)].to_parquet(f"data/bat_w{w}.parquet", index=False)
    print(" worker", w, wt)
PY
cd /workspace/INT-ACT
for w in $(seq 0 $((NW-1))); do
  /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
    --int-act-root /workspace/INT-ACT \
    --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
    --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
    --phrases /workspace/phrase-rl/data/bat_w$w.parquet \
    --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
    --repeats 12 \
    --out /workspace/phrase-rl/data/battery_out_h${HALF}_w$w.parquet > /workspace/bat_w$w.log 2>&1 &
done
wait
cd /workspace/phrase-rl
/workspace/INT-ACT/.venv/bin/python - <<PY
import glob
import pandas as pd
d = pd.concat([pd.read_parquet(f) for f in glob.glob("data/battery_out_h${HALF}_w*.parquet")], ignore_index=True)
d.to_parquet("data/rollouts_battery_half$HALF.parquet", index=False)
print(len(d), "eps merged")
PY
mkdir -p results/overnight/raw
cp -f data/rollouts_battery_half$HALF.parquet results/overnight/raw/
git add results/overnight/raw/ \
  && git -c user.name=batshard -c user.email=pod@runpod commit -m "battery half $HALF x12 [pod]" \
  && git -c user.name=batshard -c user.email=pod@runpod pull --rebase --no-edit \
  && git push
echo "BATTERY-HALF-$HALF-DONE"
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_POD_ID=')" || true
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_API_KEY=')" || true
runpodctl config --apiKey "$RUNPOD_API_KEY" 2>&1 | tail -1 || true
sleep 30
runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 | tee -a /workspace/selfstop.log
