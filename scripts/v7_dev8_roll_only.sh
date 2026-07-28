#!/usr/bin/env bash
# Roll-only half of the split dev8 pipeline: waits (via git pull) for staged
# phrase parquets from v7_dev8_gen_only.sh, rolls greedy+sampled on the probe8
# grid (24 layouts x 1 rep each), appends the merged record to the jsonl, pushes.
#   STEPS="0280 0300 0320 0340" GTAG=dev8late DEV8_OUT=v7_dev8_backfill.jsonl \
#     DEV8_TAG=dev8late bash scripts/v7_dev8_roll_only.sh
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
GEN=${MERGE_PY:-/workspace/phrase-rl/.venv-gen/bin/python}
command -v "$GEN" >/dev/null 2>&1 || [ -x "$GEN" ] || GEN=/workspace/INT-ACT/.venv/bin/python
VLA=/workspace/INT-ACT/.venv/bin/python
ROLL=/workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py
CFG=config/experiment/simpler/pi0_finetune_bridge_ev.yaml
CKPT=juexzz/INTACT-pi0-finetune-rephrase-bridge
TAG=${GTAG:-dev8}
OUT=results/analysis/${DEV8_OUT:-v7_dev8_backfill.jsonl}
mark() { echo "[rollonly $(date -u +%H:%M)] $*" | tee -a /workspace/dev8roll.log; }
wait_idle() { while pgrep -f "[p]hase0c_rollout" >/dev/null; do sleep 240; done; }

for s in ${STEPS:?set STEPS}; do
  step=$((10#$s))
  grep -q "\"step\": $step," $OUT 2>/dev/null && { mark "skip $s"; continue; }
  g=results/phrase_artifacts/dev8q_g_${TAG}_$s.parquet
  until [ -f "$g" ]; do timeout 90 git pull -q 2>/dev/null; [ -f "$g" ] || sleep 180; done
  wait_idle
  mark "roll $s greedy(192)+sampled(192) [$TAG]"
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases /workspace/phrase-rl/$g --episode-ids $(seq 0 23) --repeats 1 \
    --out /workspace/phrase-rl/data/dev8_g_out.parquet > /workspace/dev8_gr_$s.log 2>&1 &
  gp=$!
  sleep 45
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases /workspace/phrase-rl/results/phrase_artifacts/dev8q_s_${TAG}_$s.parquet --episode-ids $(seq 0 23) --repeats 1 \
    --out /workspace/phrase-rl/data/dev8_s_out.parquet > /workspace/dev8_sr_$s.log 2>&1 &
  sp=$!
  fail=0; wait $gp || fail=1; wait $sp || fail=1
  cd /workspace/phrase-rl
  [ $fail = 1 ] && { mark "ROLL FAIL $s"; continue; }
  STEP=$step DEV8_TAG=${DEV8_TAG:-$TAG} DEV8_OUT=${DEV8_OUT:-v7_dev8_backfill.jsonl} $GEN - <<'PYEOF' || { mark "MERGE FAIL $s"; continue; }
import json, os
import pandas as pd
g = pd.read_parquet("data/dev8_g_out.parquet")
s_ = pd.read_parquet("data/dev8_s_out.parquet")
rec = {"step": int(os.environ["STEP"]), "probe": os.environ.get("DEV8_TAG", "dev8"),
       "n": int(len(g)), "pooled": round(float(g.success.mean()*100), 2),
       "per_task": {t: round(float(x.success.mean()*100), 1) for t, x in g.groupby("task")},
       "sampled_n": int(len(s_)), "sampled_pooled": round(float(s_.success.mean()*100), 2)}
with open("results/analysis/" + os.environ.get("DEV8_OUT", "v7_dev8_backfill.jsonl"), "a") as f:
    f.write(json.dumps(rec) + "\n")
print("dev8 step", rec["step"], "greedy", rec["pooled"], "sampled", rec["sampled_pooled"])
PYEOF
  timeout 300 bash -c "git add $OUT && git commit -q -m 'v7 dev8 $TAG step $step [pod7]' && git pull -q --rebase && git push -q" || mark "PUSH-DEFERRED $s"
  mark "DONE $s"
done
mark "ROLL-ONLY-COMPLETE ($TAG)"
