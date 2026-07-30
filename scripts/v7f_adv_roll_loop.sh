#!/usr/bin/env bash
# v7f REPAIR eval, roll side (any render pod): polls git for staged v7fadv phrase
# parquets, rolls each unrolled step (greedy-only, 192 eps), appends to
# results/analysis/v7f_adv_curve.jsonl, pushes.
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
VLA=/workspace/INT-ACT/.venv/bin/python
MERGE=${MERGE_PY:-/workspace/INT-ACT/.venv/bin/python}
ROLL=/workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py
CFG=config/experiment/simpler/pi0_finetune_bridge_ev.yaml
CKPT=juexzz/INTACT-pi0-finetune-rephrase-bridge
OUT=results/analysis/v7f_adv_curve.jsonl
mark() { echo "[v7fadv-roll $(date -u +%H:%M)] $*" | tee -a /workspace/v7fadv_roll.log; }
wait_idle() { while pgrep -f "[p]hase0c_rollout" >/dev/null; do sleep 240; done; }

while true; do
  timeout 120 git pull -q 2>/dev/null
  for g in $(ls results/phrase_artifacts/dev8q_g_v7fadv_*.parquet 2>/dev/null); do
    s=$(basename "$g" | sed 's/.*_\([0-9]*\)\.parquet/\1/')
    step=$((10#$s))
    grep -q "\"step\": $step," $OUT 2>/dev/null && continue
    wait_idle
    mark "roll v7fadv $s greedy(192)"
    rm -f data/dev8_g_out.parquet
    cd /workspace/INT-ACT
    $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
      --phrases /workspace/phrase-rl/$g --episode-ids $(seq 0 23) --repeats ${REPEATS:-2} \
      --out /workspace/phrase-rl/data/dev8_g_out.parquet > /workspace/v7fadv_gr_$s.log 2>&1
    rc=$?
    cd /workspace/phrase-rl
    [ $rc != 0 ] && { mark "ROLL FAIL $s"; continue; }
    STEP=$step $MERGE - <<'PYEOF' || { mark "MERGE FAIL $s"; continue; }
import json, os
import pandas as pd
g = pd.read_parquet("data/dev8_g_out.parquet")
rec = {"step": int(os.environ["STEP"]), "probe": "v7f_adv",
       "n": int(len(g)), "pooled": round(float(g.success.mean()*100), 2),
       "per_task": {t: round(float(x.success.mean()*100), 1) for t, x in g.groupby("task")}}
with open("results/analysis/v7f_adv_curve.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("v7fadv step", rec["step"], "greedy", rec["pooled"])
PYEOF
    timeout 300 bash -c "git add $OUT && git commit -q -m 'v7f adv rollout step $step [pod]' && git pull -q --rebase && git push -q" || mark "PUSH-DEFERRED $s"
    mark "DONE $s"
  done
  sleep 600
done
