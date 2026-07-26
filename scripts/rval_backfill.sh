#!/usr/bin/env bash
# Comprehensive rollout-val BACKFILL from archived v7 adapters (user 2026-07-26:
# "backfill everything"). For each snapshot, regenerate greedy + sampled phrases
# from the weights and roll both at equal 192-ep budgets:
#   greedy  : 4 tasks x 24 layouts x 2 reps of greedy phrase        = 192
#   sampled : 4 tasks x 24 layouts x 1 rep of 2 samples             = 192
# Writes results/analysis/v7_curve_backfill.jsonl (own file; chart merges with the
# live curve, preferring backfill per step). Resumable (skips steps already there).
# STEP_FILTER (grep -E) shards checkpoints across pods.
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl
GEN=/workspace/phrase-rl/.venv-gen/bin/python
VLA=/workspace/INT-ACT/.venv/bin/python
ROLL=/workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py
CFG=config/experiment/simpler/pi0_finetune_bridge_ev.yaml
CKPT=juexzz/INTACT-pi0-finetune-rephrase-bridge
ADAPTERS=/workspace/v7_adapters
CURVE=results/analysis/v7_curve_backfill.jsonl
NSAMP=${RVAL_SAMPLES:-2}
mark() { echo "[bfill $(date -u +%H:%M)] $*" | tee -a /workspace/backfill.log; }

roll_one() {  # <phrases> <repeats> <out>
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases "$1" --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
    --repeats "$2" --out "$3" > "${3%.parquet}.log" 2>&1
  local rc=$?; cd /workspace/phrase-rl; return $rc
}

mark "start (filter='${STEP_FILTER:-all}', n_samples=$NSAMP)"
for d in $(ls -d $ADAPTERS/step_* | sort); do
  s=$(basename "$d" | sed 's/step_//')
  step=$((10#$s))
  [ "$step" = 0 ] && continue                       # step 0 = untrained base
  [[ "$s" =~ ${STEP_FILTER:-.} ]] || continue
  [ -f "$d/adapter_model.safetensors" ] || { mark "SKIP $s (no adapter yet)"; continue; }
  if [ -f "$CURVE" ] && grep -q "\"step\": $step," "$CURVE" 2>/dev/null; then
    mark "skip $s (already in curve)"; continue
  fi
  mark "checkpoint $s: generating phrases"
  $GEN scripts/gen_ckpt_phrases.py "$d" "$NSAMP" > /workspace/bf_gen_$s.log 2>&1 \
    || { mark "GEN FAILED $s (see bf_gen_$s.log)"; continue; }
  rm -f data/rval_greedy_out.parquet data/rval_sampled_out.parquet
  mark "checkpoint $s: rolling greedy(192)+sampled(192)"
  roll_one /workspace/phrase-rl/data/rval_greedy.parquet 2 /workspace/phrase-rl/data/rval_greedy_out.parquet & gp=$!
  sleep 45
  roll_one /workspace/phrase-rl/data/rval_sampled.parquet 1 /workspace/phrase-rl/data/rval_sampled_out.parquet & sp=$!
  wfail=0; wait $gp || wfail=1; wait $sp || wfail=1
  [ $wfail = 1 ] && { mark "ROLL FAILED $s"; continue; }
  STEP=$step $GEN - <<'PYEOF' || { mark "MERGE FAILED $s"; continue; }
import json, os
import pandas as pd
g = pd.read_parquet("data/rval_greedy_out.parquet")
s = pd.read_parquet("data/rval_sampled_out.parquet")
rec = {"step": int(os.environ["STEP"]), "src": "backfill",
       "n": int(len(g)), "pooled": round(float(g.success.mean() * 100), 2),
       "per_task": {t: round(float(x.success.mean() * 100), 1) for t, x in g.groupby("task")},
       "sampled_n": int(len(s)), "sampled_pooled": round(float(s.success.mean() * 100), 2),
       "sampled_per_task": {t: round(float(x.success.mean() * 100), 1) for t, x in s.groupby("task")}}
with open("results/analysis/v7_curve_backfill.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("step", rec["step"], "greedy", rec["pooled"], "sampled", rec["sampled_pooled"])
PYEOF
  for i in 1 2 3; do
    git add "$CURVE" && git commit -q -m "v7 backfill curve step $step (greedy+sampled) [pod]" \
      && git pull -q --rebase && git push -q && break
    sleep $((i * 20))
  done
  mark "DONE $s"
done
mark "BACKFILL-COMPLETE"
