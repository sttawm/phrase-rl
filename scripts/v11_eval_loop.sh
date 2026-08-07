#!/bin/bash
# v11 single-pod checkpoint evaluator (A27): generate prompt-B+ phrases from each
# archived v11 checkpoint and roll them on the NATURAL-input val probe (primary
# metric). One pod does both roles — no claims, no fan-out.
#   STRIDE=10 bash scripts/v11_eval_loop.sh
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
GEN=/workspace/phrase-rl/.venv-gen/bin/python
VLA=/workspace/INT-ACT/.venv/bin/python
ROLL=/workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py
CFG=config/experiment/simpler/pi0_finetune_bridge_ev.yaml
CKPT=juexzz/INTACT-pi0-finetune-rephrase-bridge
STRIDE=${STRIDE:-10}
mark() { echo "[v11eval $(date -u +%H:%M)] $*" >> /workspace/v11eval.log; }

mkdir -p results/analysis/v11cells /workspace/v11_adapters

while true; do
  timeout 240 git -c rebase.autoStash=true pull -q --rebase 2>/dev/null \
    || { git rebase --abort 2>/dev/null; git reset --hard -q origin/main; }

  # next archived stride-multiple step with no cell yet
  last=$(ls results/analysis/v11cells/nat_*.json 2>/dev/null | sed 's/.*nat_0*\([0-9][0-9]*\)\.json/\1/' | sort -n | tail -1)
  last=${last:-0}
  next=$(ls results/checkpoints/archive/ 2>/dev/null | grep -oE "v11_step_[0-9]+" | grep -oE "[0-9]+$" \
         | sort -n -u | awk -v t=$last -v s=$STRIDE '$1 > t && $1 % s == 0' | head -1)
  if [ -z "$next" ]; then mark "idle (last cell=$last, no new archive)"; sleep 600; continue; fi
  s=$(printf "%04d" $((10#$next)))

  d=/workspace/v11_adapters/step_$s
  if [ ! -f "$d/adapter_model.safetensors" ]; then
    mkdir -p /tmp/v11ck && rm -rf /tmp/v11ck/*
    cat results/checkpoints/archive/v11_step_$s.tgz.part-* 2>/dev/null | tar xzf - -C /tmp/v11ck 2>/dev/null
    src=$(find /tmp/v11ck -name adapter_model.safetensors | head -1)
    [ -n "$src" ] && { mkdir -p "$d"; cp "$(dirname $src)"/* "$d"/; } \
      || { mark "archive incomplete for $s"; sleep 600; continue; }
  fi

  mark "gen nat $s (prompt B+)"
  GEN_TAG="" PROMPT_FAMILY=bplus \
    PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8_nat.parquet \
    PROBE_TRACES=results/phrase_artifacts/traces_probe8.parquet \
    $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/v11gen_$s.log 2>&1 \
    || { mark "GEN FAIL $s"; sleep 300; continue; }
  cp data/rval_greedy.parquet results/phrase_artifacts/dev11q_g_nat_$s.parquet

  mark "roll nat $s greedy(192)"
  rm -f data/dev11_g_out.parquet
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases /workspace/phrase-rl/results/phrase_artifacts/dev11q_g_nat_$s.parquet \
    --episode-ids $(seq 0 23) --repeats 1 \
    --out /workspace/phrase-rl/data/dev11_g_out.parquet > /workspace/v11roll_$s.log 2>&1
  rc=$?
  cd /workspace/phrase-rl
  [ $rc != 0 ] && { mark "ROLL FAIL $s"; sleep 300; continue; }

  STEP=$s CELLF=results/analysis/v11cells/nat_$s.json $VLA - <<'PYEOF' || { mark "MERGE FAIL $s"; continue; }
import json, os
import pandas as pd
g = pd.read_parquet("data/dev11_g_out.parquet")
rec = {"step": int(os.environ["STEP"]), "probe": "v11_nat",
       "n": int(len(g)), "pooled": round(float(g.success.mean() * 100), 2),
       "per_task": {t: round(float(x.success.mean() * 100), 1) for t, x in g.groupby("task")}}
with open(os.environ["CELLF"], "w") as f:
    json.dump(rec, f)
print("v11_nat step", rec["step"], "pooled", rec["pooled"])
PYEOF

  for i in 1 2 3; do
    timeout 300 bash -c "git add results/analysis/v11cells/nat_$s.json results/phrase_artifacts/dev11q_g_nat_$s.parquet && git commit -q -m 'v11 nat cell $s [e4]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" \
      && { mark "DONE nat $s"; break; }
    git rebase --abort 2>/dev/null; sleep $((30 * i))
  done
done
