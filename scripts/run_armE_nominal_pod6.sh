#!/usr/bin/env bash
# Arm E NOMINAL leg on pod6 (parallelizes arm E: pod8 keeps the ERT leg).
# One-shot: roll 12x24x12, merge, x12 copy, push, 1-task drift re-anchor
# (prereg anchor clause — pod6 is new to sealed rolling), then hand pod6 back
# to the v8 eval worker. Pod8's runner gets killed at its DONE-ert event so its
# own nominal iteration never double-rolls.
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
mark() { echo "[armEnom $(date -u +%H:%M)] $*" | tee -a /workspace/armE_nominal.log; }

timeout 120 git -c rebase.autoStash=true pull -q 2>/dev/null
if grep -q '"arm": "rules_v4_nominal"' $OUT 2>/dev/null; then mark "already merged — abort"; exit 0; fi
mark "roll rules_v4_nominal (12x24x12)"
rm -f data/armE_nom_out.parquet
cd /workspace/INT-ACT
$VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
  --phrases /workspace/phrase-rl/results/sealed/ph_sealed_rules_v4_nominal.parquet \
  --episode-ids $(seq 0 23) --repeats 12 \
  --out /workspace/phrase-rl/data/armE_nom_out.parquet > /workspace/armE_nom_roll.log 2>&1
rc=$?
cd /workspace/phrase-rl
[ $rc != 0 ] && { mark "ROLL FAIL nominal"; exit 1; }
/workspace/INT-ACT/.venv/bin/python - <<'PYEOF' || { mark "MERGE FAIL nominal"; exit 1; }
import json
import pandas as pd
g = pd.read_parquet("data/armE_nom_out.parquet")
assert len(g) == 3456, f"n={len(g)} != 3456 (contamination tell)"
rec = {"arm": "rules_v4_nominal", "pod": "pod6", "n": int(len(g)),
       "pooled": round(float(g.success.mean()*100), 2),
       "per_task": {t: round(float(x.success.mean()*100), 1) for t, x in g.groupby("task")}}
with open("results/analysis/sealed_rules_v4.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("rules_v4_nominal", rec["pooled"])
PYEOF
cp data/armE_nom_out.parquet results/sealed/rules_v4_nominal_x12.parquet
timeout 300 bash -c "git add $OUT results/sealed/rules_v4_nominal_x12.parquet && git commit -q -m 'arm E: rules_v4_nominal x12 [pod6]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" || mark "PUSH-DEFERRED nominal"
mark "DONE nominal"

mark "drift re-anchor (cube_on_plate originals, 24x3)"
/workspace/INT-ACT/.venv/bin/python - <<'PYEOF'
import pandas as pd
a = pd.read_parquet("results/sealed/ph_sealed_anchors.parquet")
a[a.task == "widowx_cube_on_plate_clean"].to_parquet("/workspace/anchor_one.parquet", index=False)
PYEOF
rm -f data/anchor_drift_out.parquet
cd /workspace/INT-ACT
$VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
  --phrases /workspace/anchor_one.parquet --episode-ids $(seq 0 23) --repeats 3 \
  --out /workspace/phrase-rl/data/anchor_drift_out.parquet > /workspace/anchor_drift6.log 2>&1
cd /workspace/phrase-rl
/workspace/INT-ACT/.venv/bin/python - <<'PYEOF' || mark "ANCHOR MERGE FAIL"
import json
import pandas as pd
g = pd.read_parquet("data/anchor_drift_out.parquet")
rec = {"arm": "drift_anchor_cube_on_plate", "pod": "pod6", "n": int(len(g)),
       "pooled": round(float(g.success.mean()*100), 2)}
with open("results/analysis/sealed_rules_v4.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("drift anchor pod6", rec["pooled"])
PYEOF
timeout 300 bash -c "git add $OUT && git commit -q -m 'pod6 drift re-anchor [pod6]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" || mark "PUSH-DEFERRED anchor"
mark "ARMENOM-COMPLETE — handing pod6 back to v8 worker"
exec bash scripts/v8_gen_roll_loop.sh
