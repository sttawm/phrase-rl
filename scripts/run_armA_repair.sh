#!/usr/bin/env bash
# Arm A REPAIR leg on pod7 (parallelizes the sealed closing leg: pod5 keeps the
# polish roll, pod7 takes repair — Amendment 8). Identical protocol to pod5's leg:
# ph_sealed_v7a120_repair.parquet x 24 layouts x 12 reps on the standard executor.
# Afterwards: a 1-task originals re-anchor (cross-pod drift cell, prereg anchor
# clause), then exec the unified eval roller (v8 stride-10 duty).
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
VLA=/workspace/INT-ACT/.venv/bin/python
ROLL=/workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py
CFG=config/experiment/simpler/pi0_finetune_bridge_ev.yaml
CKPT=juexzz/INTACT-pi0-finetune-rephrase-bridge
OUT=results/analysis/sealed_v7a120.jsonl
mark() { echo "[armArep $(date -u +%H:%M)] $*" | tee -a /workspace/armA_repair.log; }

timeout 120 git -c rebase.autoStash=true pull -q 2>/dev/null
if ! grep -q '"arm": "sealed_v7a120_repair"' $OUT 2>/dev/null; then
  mark "roll arm A REPAIR (12x24x12)"
  rm -f data/armA_rep_out.parquet
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases /workspace/phrase-rl/results/sealed/ph_sealed_v7a120_repair.parquet \
    --episode-ids $(seq 0 23) --repeats 12 \
    --out /workspace/phrase-rl/data/armA_rep_out.parquet > /workspace/armA_rep_roll.log 2>&1
  rc=$?
  cd /workspace/phrase-rl
  if [ $rc = 0 ]; then
    /workspace/INT-ACT/.venv/bin/python - <<'PYEOF' && mark "REPAIR MERGED" || mark "MERGE FAIL repair"
import json
import pandas as pd
g = pd.read_parquet("data/armA_rep_out.parquet")
assert len(g) == 3456, f"n={len(g)} != 3456 (contamination tell)"
rec = {"arm": "sealed_v7a120_repair", "pod": "pod7", "n": int(len(g)),
       "pooled": round(float(g.success.mean()*100), 2),
       "per_task": {t: round(float(x.success.mean()*100), 1) for t, x in g.groupby("task")}}
with open("results/analysis/sealed_v7a120.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("armA repair", rec["pooled"])
PYEOF
    cp data/armA_rep_out.parquet results/sealed/v7a120_repair_x12.parquet
    timeout 300 bash -c "git add $OUT results/sealed/v7a120_repair_x12.parquet && git commit -q -m 'arm A repair leg x12 [pod7]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" || mark "PUSH-DEFERRED repair"
    mark "DONE armA repair"
  else
    mark "ROLL FAIL armA repair"
  fi
else
  mark "repair already merged — skip"
fi

# cross-pod drift re-anchor: 1 task x 24 x 3 originals (prereg anchor clause)
mark "drift re-anchor (cube_on_plate originals, 24x3)"
rm -f data/anchor_drift_out.parquet
/workspace/INT-ACT/.venv/bin/python - <<'PYEOF'
import pandas as pd
a = pd.read_parquet("results/sealed/ph_sealed_anchors.parquet")
a[a.task == "widowx_cube_on_plate_clean"].to_parquet("/workspace/anchor_one.parquet", index=False)
PYEOF
cd /workspace/INT-ACT
$VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
  --phrases /workspace/anchor_one.parquet --episode-ids $(seq 0 23) --repeats 3 \
  --out /workspace/phrase-rl/data/anchor_drift_out.parquet > /workspace/anchor_drift.log 2>&1
cd /workspace/phrase-rl
/workspace/INT-ACT/.venv/bin/python - <<'PYEOF' || mark "ANCHOR MERGE FAIL"
import json
import pandas as pd
g = pd.read_parquet("data/anchor_drift_out.parquet")
rec = {"arm": "drift_anchor_cube_on_plate", "pod": "pod7", "n": int(len(g)),
       "pooled": round(float(g.success.mean()*100), 2)}
with open("results/analysis/sealed_v7a120.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("drift anchor", rec["pooled"])
PYEOF
timeout 300 bash -c "git add $OUT && git commit -q -m 'pod7 drift re-anchor [pod7]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" || mark "PUSH-DEFERRED anchor"
mark "ARMAREP-COMPLETE — handing off to unified eval roller"
exec bash scripts/eval_roll_loop.sh
