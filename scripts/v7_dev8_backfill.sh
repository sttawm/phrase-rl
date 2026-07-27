#!/usr/bin/env bash
# Dev-8 rollout backfill for selected v7 checkpoints (user 2026-07-27): greedy +
# 1-sample per task on the 8-task probe grid, 24 layouts x 1 rep each (192+192).
# Writes results/analysis/v7_dev8_backfill.jsonl. Yields to running phase0c.
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
export PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8.parquet
export PROBE_TRACES=results/phrase_artifacts/traces_probe8.parquet
cd /workspace/phrase-rl
GEN=/workspace/phrase-rl/.venv-gen/bin/python
VLA=/workspace/INT-ACT/.venv/bin/python
ROLL=/workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py
CFG=config/experiment/simpler/pi0_finetune_bridge_ev.yaml
CKPT=juexzz/INTACT-pi0-finetune-rephrase-bridge
OUT=results/analysis/v7_dev8_backfill.jsonl
mark() { echo "[dev8 $(date -u +%H:%M)] $*" | tee -a /workspace/dev8.log; }
wait_idle() { while pgrep -f "[p]hase0c_rollout" >/dev/null; do sleep 240; done; }

for s in ${STEPS:-0120 0140 0200 0260}; do
  step=$((10#$s))
  grep -q "\"step\": $step," $OUT 2>/dev/null && { mark "skip $s"; continue; }
  d=/workspace/v7_adapters/step_$s
  [ -f "$d/adapter_model.safetensors" ] || { mark "no adapter $s"; continue; }
  wait_idle
  mark "gen $s (probe8)"
  $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/dev8_gen_$s.log 2>&1 || { mark "GEN FAIL $s"; continue; }
  rm -f data/dev8_g.parquet data/dev8_s.parquet
  cp data/rval_greedy.parquet data/dev8_g.parquet
  cp data/rval_sampled.parquet data/dev8_s.parquet
  wait_idle
  mark "roll $s greedy(192)+sampled(192)"
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases /workspace/phrase-rl/data/dev8_g.parquet --episode-ids $(seq 0 23) --repeats 1 \
    --out /workspace/phrase-rl/data/dev8_g_out.parquet > /workspace/dev8_gr_$s.log 2>&1 &
  gp=$!
  sleep 45
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases /workspace/phrase-rl/data/dev8_s.parquet --episode-ids $(seq 0 23) --repeats 1 \
    --out /workspace/phrase-rl/data/dev8_s_out.parquet > /workspace/dev8_sr_$s.log 2>&1 &
  sp=$!
  fail=0; wait $gp || fail=1; wait $sp || fail=1
  cd /workspace/phrase-rl
  [ $fail = 1 ] && { mark "ROLL FAIL $s"; continue; }
  STEP=$step $GEN - <<'PYEOF' || { mark "MERGE FAIL $s"; continue; }
import json, os
import pandas as pd
g = pd.read_parquet("data/dev8_g_out.parquet")
s_ = pd.read_parquet("data/dev8_s_out.parquet")
rec = {"step": int(os.environ["STEP"]), "probe": "dev8",
       "n": int(len(g)), "pooled": round(float(g.success.mean()*100), 2),
       "per_task": {t: round(float(x.success.mean()*100), 1) for t, x in g.groupby("task")},
       "sampled_n": int(len(s_)), "sampled_pooled": round(float(s_.success.mean()*100), 2)}
with open("results/analysis/v7_dev8_backfill.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("dev8 step", rec["step"], "greedy", rec["pooled"], "sampled", rec["sampled_pooled"])
PYEOF
  timeout 300 bash -c "git add $OUT && git commit -q -m 'v7 dev8 backfill step $step [pod]' && git pull -q --rebase && git push -q" || mark "PUSH-DEFERRED $s"
  mark "DONE $s"
done
mark "DEV8-BACKFILL-COMPLETE"
