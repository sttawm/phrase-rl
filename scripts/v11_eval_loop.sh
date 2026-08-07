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

# --- baselines on the SAME probe/protocol, produced once before checkpoint cells:
#   nat_0000  = frozen base Qwen + prompt B+ (RL step 0)
#   nat_pass  = the natural inputs rolled with NO rephraser (passthrough)
roll_and_cell() { # $1 phrases parquet (repo-relative), $2 cell name
  rm -f data/dev11_g_out.parquet
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases /workspace/phrase-rl/$1 --episode-ids $(seq 0 23) --repeats 1 \
    --out /workspace/phrase-rl/data/dev11_g_out.parquet > /workspace/v11roll_$2.log 2>&1
  rc=$?; cd /workspace/phrase-rl
  [ $rc != 0 ] && { mark "ROLL FAIL $2"; return 1; }
  STEP=$2 CELLF=results/analysis/v11cells/$2.json $VLA - <<'PYEOF'
import json, os
import pandas as pd
g = pd.read_parquet("data/dev11_g_out.parquet")
rec = {"step": os.environ["STEP"], "probe": "v11_nat",
       "n": int(len(g)), "pooled": round(float(g.success.mean() * 100), 2),
       "per_task": {t: round(float(x.success.mean() * 100), 1) for t, x in g.groupby("task")}}
json.dump(rec, open(os.environ["CELLF"], "w"))
print(os.environ["STEP"], "pooled", rec["pooled"])
PYEOF
  for i in 1 2 3; do
    timeout 300 bash -c "git add results/analysis/v11cells/$2.json && git commit -q -m 'v11 baseline cell $2 [e4]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" && { mark "DONE $2"; return 0; }
    git rebase --abort 2>/dev/null; sleep 30
  done
}

if [ ! -f results/analysis/v11cells/nat24_pass.json ]; then
  mark "baseline: passthrough (no rephraser)"
  $GEN - <<'PYEOF'
import pandas as pd
c = pd.read_parquet("results/phrase_artifacts/contexts_probe8_nat24.parquet")
pd.DataFrame({"task": c.task, "arm": "passthrough", "phrase": c.instruction,
              "instruction": c.instruction, "episode_id": c.episode_id}).to_parquet(
    "results/phrase_artifacts/dev11q_g_nat24_pass.parquet", index=False)
PYEOF
  roll_and_cell results/phrase_artifacts/dev11q_g_nat24_pass.parquet nat24_pass
fi

if [ ! -f results/analysis/v11cells/nat24_0000.json ]; then
  mark "baseline: BASE qwen + prompt B+ (step 0)"
  GEN_TAG="" PROMPT_FAMILY=bplus \
    PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8_nat24.parquet \
    PROBE_TRACES=results/phrase_artifacts/traces_probe8.parquet \
    $GEN scripts/gen_ckpt_phrases.py BASE 1 > /workspace/v11gen_base.log 2>&1 \
    && cp data/rval_greedy.parquet results/phrase_artifacts/dev11q_g_nat24_0000.parquet \
    && roll_and_cell results/phrase_artifacts/dev11q_g_nat24_0000.parquet nat24_0000 \
    || mark "BASE GEN FAIL"
fi

while true; do
  timeout 240 git -c rebase.autoStash=true pull -q --rebase 2>/dev/null \
    || { git rebase --abort 2>/dev/null; git reset --hard -q origin/main; }

  # next archived stride-multiple step with no cell yet
  last=$(ls results/analysis/v11cells/nat24_*.json 2>/dev/null | sed 's/.*nat24_0*\([0-9][0-9]*\)\.json/\1/' | sort -n | tail -1)
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
    PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8_nat24.parquet \
    PROBE_TRACES=results/phrase_artifacts/traces_probe8.parquet \
    $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/v11gen_$s.log 2>&1 \
    || { mark "GEN FAIL $s"; sleep 300; continue; }
  cp data/rval_greedy.parquet results/phrase_artifacts/dev11q_g_nat24_$s.parquet

  mark "roll nat $s greedy(192)"
  rm -f data/dev11_g_out.parquet
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases /workspace/phrase-rl/results/phrase_artifacts/dev11q_g_nat24_$s.parquet \
    --episode-ids $(seq 0 23) --repeats 1 \
    --out /workspace/phrase-rl/data/dev11_g_out.parquet > /workspace/v11roll_$s.log 2>&1
  rc=$?
  cd /workspace/phrase-rl
  [ $rc != 0 ] && { mark "ROLL FAIL $s"; sleep 300; continue; }

  STEP=$s CELLF=results/analysis/v11cells/nat24_$s.json $VLA - <<'PYEOF' || { mark "MERGE FAIL $s"; continue; }
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
    timeout 300 bash -c "git add results/analysis/v11cells/nat24_$s.json results/phrase_artifacts/dev11q_g_nat24_$s.parquet && git commit -q -m 'v11 nat24 cell $s [e4]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" \
      && { mark "DONE nat $s"; break; }
    git rebase --abort 2>/dev/null; sleep $((30 * i))
  done
done
