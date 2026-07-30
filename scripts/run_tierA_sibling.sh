#!/usr/bin/env bash
# Tier-A generalization leg (PREREG): frozen sealed text arms rolled on the
# NON-paraphrase-augmented sibling executor (juexzz/INTACT-pi0-finetune-bridge).
# 3 arms x 12 tasks x 24 layouts x 3 reps. Output: results/analysis/tierA_sibling.jsonl
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl
VLA=/workspace/INT-ACT/.venv/bin/python
ROLL=/workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py
CFG=config/experiment/simpler/pi0_finetune_bridge_ev.yaml
SIB=juexzz/INTACT-pi0-finetune-bridge
OUT=results/analysis/tierA_sibling.jsonl
mark() { echo "[tierA $(date -u +%H:%M)] $*" | tee -a /workspace/tierA.log; }

for arm in anchors rules_v3_gemini_pro oracle_confirmed; do
  grep -q "\"arm\": \"$arm\"" $OUT 2>/dev/null && { mark "skip $arm"; continue; }
  ph=results/sealed/ph_sealed_$arm.parquet
  [ -f "$ph" ] || { mark "MISSING $ph"; continue; }
  mark "roll $arm (12x24x3) on sibling ckpt"
  rm -f data/tierA_out.parquet
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $SIB \
    --phrases /workspace/phrase-rl/$ph --episode-ids $(seq 0 23) --repeats 3 \
    --out /workspace/phrase-rl/data/tierA_out.parquet > /workspace/tierA_$arm.log 2>&1
  rc=$?
  cd /workspace/phrase-rl
  [ $rc != 0 ] && { mark "ROLL FAIL $arm"; continue; }
  ARM=$arm /workspace/INT-ACT/.venv/bin/python - <<'PYEOF' || { mark "MERGE FAIL $arm"; continue; }
import json, os
import pandas as pd
g = pd.read_parquet("data/tierA_out.parquet")
rec = {"arm": os.environ["ARM"], "executor": "INTACT-pi0-finetune-bridge",
       "n": int(len(g)), "pooled": round(float(g.success.mean()*100), 2),
       "per_task": {t: round(float(x.success.mean()*100), 1) for t, x in g.groupby("task")}}
with open("results/analysis/tierA_sibling.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("tierA", rec["arm"], rec["pooled"])
PYEOF
  timeout 300 bash -c "git add $OUT && git commit -q -m 'tierA sibling: $arm [pod8]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" || mark "PUSH-DEFERRED $arm"
  mark "DONE $arm"
done
mark "TIERA-COMPLETE"
