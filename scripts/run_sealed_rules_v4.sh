#!/usr/bin/env bash
# Arm E sealed rolls (PREREG Amendment 7): rules-v4 phrases, both conditions,
# 12 tasks x 24 layouts x 12 reps each, on the standard executor. Runs on pod8
# AFTER the Tier-A leg (waits on the TIERA-COMPLETE log marker, not a process —
# pgrep self-match footgun). Output: results/analysis/sealed_rules_v4.jsonl.
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
VLA=/workspace/INT-ACT/.venv/bin/python
ROLL=/workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py
CFG=config/experiment/simpler/pi0_finetune_bridge_ev.yaml
CKPT=juexzz/INTACT-pi0-finetune-rephrase-bridge
OUT=results/analysis/sealed_rules_v4.jsonl
mark() { echo "[rulesv4 $(date -u +%H:%M)] $*" | tee -a /workspace/rules_v4.log; }

mark "waiting for TIERA-COMPLETE"
until grep -q "TIERA-COMPLETE" /workspace/tierA.log 2>/dev/null; do sleep 600; done
mark "tierA done — starting arm E"

for cond in ert nominal; do
  timeout 120 git -c rebase.autoStash=true pull -q 2>/dev/null
  grep -q "\"arm\": \"rules_v4_$cond\"" $OUT 2>/dev/null && { mark "skip $cond"; continue; }
  ph=results/sealed/ph_sealed_rules_v4_$cond.parquet
  [ -f "$ph" ] || { mark "MISSING $ph — will retry next pass"; sleep 1800; timeout 120 git pull -q; [ -f "$ph" ] || { mark "STILL MISSING $ph"; continue; }; }
  mark "roll rules_v4_$cond (12x24x12)"
  rm -f data/rules_v4_out.parquet
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases /workspace/phrase-rl/$ph --episode-ids $(seq 0 23) --repeats 12 \
    --out /workspace/phrase-rl/data/rules_v4_out.parquet > /workspace/rules_v4_$cond.log 2>&1
  rc=$?
  cd /workspace/phrase-rl
  [ $rc != 0 ] && { mark "ROLL FAIL $cond"; continue; }
  CONDN=$cond /workspace/INT-ACT/.venv/bin/python - <<'PYEOF' || { mark "MERGE FAIL $cond"; continue; }
import json, os
import pandas as pd
g = pd.read_parquet("data/rules_v4_out.parquet")
assert len(g) == 3456, f"n={len(g)} != 3456 (contamination tell)"
rec = {"arm": "rules_v4_" + os.environ["CONDN"], "n": int(len(g)),
       "pooled": round(float(g.success.mean()*100), 2),
       "per_task": {t: round(float(x.success.mean()*100), 1) for t, x in g.groupby("task")}}
with open("results/analysis/sealed_rules_v4.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("rules_v4", rec["arm"], rec["pooled"])
PYEOF
  cp data/rules_v4_out.parquet results/sealed/rules_v4_${cond}_x12.parquet
  timeout 300 bash -c "git add $OUT results/sealed/rules_v4_${cond}_x12.parquet && git commit -q -m 'arm E: rules_v4_$cond x12 [pod8]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" || mark "PUSH-DEFERRED $cond"
  mark "DONE $cond"
done
mark "ARM-E-COMPLETE"
