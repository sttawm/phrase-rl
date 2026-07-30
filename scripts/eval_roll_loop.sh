#!/usr/bin/env bash
# Unified checkpoint-eval roller (pod7): rolls staged phrase parquets for BOTH the
# v7f family (legacy queue) and the v8 family, oldest-step-first within each pass.
# Curve files: v7f_{adv,pol}_curve.jsonl / v8_{adv,pol}_curve.jsonl. Greedy, 192 eps
# x REPEATS (default 2). Successor of v7f_adv_roll_loop.sh (same merge contract).
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
mark() { echo "[eval-roll $(date -u +%H:%M)] $*" | tee -a /workspace/eval_roll.log; }
wait_idle() { while pgrep -f "[p]hase0c_rollout" >/dev/null; do sleep 240; done; }

while true; do
  timeout 120 git -c rebase.autoStash=true pull -q 2>/dev/null
  for g in $(ls results/phrase_artifacts/dev8q_g_v7fadv_*.parquet results/phrase_artifacts/dev8q_g_v7fpol_*.parquet \
                results/phrase_artifacts/dev8q_g_v8adv_*.parquet results/phrase_artifacts/dev8q_g_v8pol_*.parquet 2>/dev/null \
             | awk -F_ "{print \$NF, \$0}" | sort -n | cut -d" " -f2); do
    s=$(basename "$g" | sed 's/.*_\([0-9]*\)\.parquet/\1/')
    step=$((10#$s))
    case "$g" in
      *v8pol*)  OUT=results/analysis/v8_pol_curve.jsonl;  COND=v8_pol;;
      *v8adv*)  OUT=results/analysis/v8_adv_curve.jsonl;  COND=v8_adv;;
      *v7fpol*) OUT=results/analysis/v7f_pol_curve.jsonl; COND=v7f_pol;;
      *)        OUT=results/analysis/v7f_adv_curve.jsonl; COND=v7f_adv;;
    esac
    grep -q "\"step\": $step," $OUT 2>/dev/null && continue
    wait_idle
    mark "roll $COND $s greedy(192)"
    rm -f data/dev8_g_out.parquet
    cd /workspace/INT-ACT
    $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
      --phrases /workspace/phrase-rl/$g --episode-ids $(seq 0 23) --repeats ${REPEATS:-2} \
      --out /workspace/phrase-rl/data/dev8_g_out.parquet > /workspace/eval_gr_${COND}_$s.log 2>&1
    rc=$?
    cd /workspace/phrase-rl
    [ $rc != 0 ] && { mark "ROLL FAIL $COND $s"; continue; }
    STEP=$step COND=$COND OUTF=$OUT $MERGE - <<'PYEOF' || { mark "MERGE FAIL $COND $s"; continue; }
import json, os
import pandas as pd
g = pd.read_parquet("data/dev8_g_out.parquet")
rec = {"step": int(os.environ["STEP"]), "probe": os.environ["COND"],
       "n": int(len(g)), "pooled": round(float(g.success.mean()*100), 2),
       "per_task": {t: round(float(x.success.mean()*100), 1) for t, x in g.groupby("task")}}
with open(os.environ["OUTF"], "a") as f:
    f.write(json.dumps(rec) + "\n")
print(os.environ["COND"], "step", rec["step"], "greedy", rec["pooled"])
PYEOF
    timeout 300 bash -c "git add $OUT && git commit -q -m 'eval $COND step $step [pod]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" || mark "PUSH-DEFERRED $COND $s"
    mark "DONE $COND $s"
  done
  sleep 600
done
