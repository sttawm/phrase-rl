#!/usr/bin/env bash
# v9 all-in-one eval worker (pod6): per new archived checkpoint, sequentially
#   1. generate tag-free rewrites, BOTH conditions (adv = probe8_adv ERT inputs,
#      pol = probe8 nominal inputs), stage + push parquets
#   2. roll each condition greedy x 192 x REPEATS(2), merge to
#      v9_{adv,pol}_curve.jsonl, push
# Replaces the split pod6-gen / pod7-roll pipeline (fleet consolidation: one GPU
# does 30min gen + ~3.6h rolls inside the ~6h checkpoint cadence). v9 files only.
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
mark() { echo "[v9work $(date -u +%H:%M)] $*" | tee -a /workspace/v9work.log; }

roll_one() { # $1 staged parquet, $2 OUT jsonl, $3 COND label, $4 step
  grep -q "\"step\": $4," "$2" 2>/dev/null && return 0
  mark "roll $3 $4 greedy(192)"
  rm -f data/dev9_g_out.parquet
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases /workspace/phrase-rl/$1 --episode-ids $(seq 0 23) --repeats ${REPEATS:-2} \
    --out /workspace/phrase-rl/data/dev9_g_out.parquet > /workspace/v9roll_$3_$4.log 2>&1
  rc=$?
  cd /workspace/phrase-rl
  [ $rc != 0 ] && { mark "ROLL FAIL $3 $4"; return 1; }
  STEP=$4 COND=$3 OUTF=$2 $VLA - <<'PYEOF' || { mark "MERGE FAIL $3 $4"; return 1; }
import json, os
import pandas as pd
g = pd.read_parquet("data/dev9_g_out.parquet")
rec = {"step": int(os.environ["STEP"]), "probe": os.environ["COND"],
       "n": int(len(g)), "pooled": round(float(g.success.mean()*100), 2),
       "per_task": {t: round(float(x.success.mean()*100), 1) for t, x in g.groupby("task")}}
with open(os.environ["OUTF"], "a") as f:
    f.write(json.dumps(rec) + "\n")
print(os.environ["COND"], "step", rec["step"], "greedy", rec["pooled"])
PYEOF
  timeout 300 bash -c "git add $2 && git commit -q -m 'v9 eval $3 step $4 [pod6]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" || mark "PUSH-DEFERRED $3 $4"
  mark "DONE $3 $4"
}

while true; do
  timeout 120 git -c rebase.autoStash=true pull -q 2>/dev/null
  last=$(ls results/phrase_artifacts/dev9q_g_v9adv_*.parquet 2>/dev/null | sed 's/.*_0*\([0-9]*\)\.parquet/\1/' | sort -n | tail -1)
  last=${last:-0}
  next=$(ls results/checkpoints/archive/ 2>/dev/null | grep -oE "v9_step_[0-9]+" | grep -oE "[0-9]+$" | sort -n | awk -v t=$((last + STRIDE)) '$1 >= t' | head -1)
  if [ -n "$next" ]; then
    s=$(printf "%04d" $((10#$next)))
    d=/workspace/v9_adapters/step_$s
    if [ ! -f "$d/adapter_model.safetensors" ]; then
      mkdir -p /tmp/v9ck && rm -rf /tmp/v9ck/*
      cat results/checkpoints/archive/v9_step_$s.tgz.part-* 2>/dev/null | tar xzf - -C /tmp/v9ck 2>/dev/null
      src=$(find /tmp/v9ck -name adapter_model.safetensors | head -1)
      [ -n "$src" ] && { mkdir -p "$d"; cp "$(dirname $src)"/* "$d"/; } || { mark "no archive for $s yet"; sleep 600; continue; }
    fi
    mark "gen adv $s (tag-free)"
    GEN_TAG="" \
      PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8_adv.parquet \
      PROBE_TRACES=results/phrase_artifacts/traces_probe8_adv.parquet \
      $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/v9adv_gen_$s.log 2>&1 || { mark "GEN FAIL $s"; sleep 600; continue; }
    cp data/rval_greedy.parquet results/phrase_artifacts/dev9q_g_v9adv_$s.parquet
    cp data/rval_sampled.parquet results/phrase_artifacts/dev9q_s_v9adv_$s.parquet
    mark "gen polish $s (tag-free)"
    GEN_TAG="" \
      PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8.parquet \
      PROBE_TRACES=results/phrase_artifacts/traces_probe8.parquet \
      $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/v9pol_gen_$s.log 2>&1 || { mark "POL GEN FAIL $s"; }
    cp data/rval_greedy.parquet results/phrase_artifacts/dev9q_g_v9pol_$s.parquet 2>/dev/null
    cp data/rval_sampled.parquet results/phrase_artifacts/dev9q_s_v9pol_$s.parquet 2>/dev/null
    timeout 300 bash -c "git add results/phrase_artifacts/dev9q_*v9adv*.parquet results/phrase_artifacts/dev9q_*v9pol*.parquet && git commit -q -m 'v9 eval phrases $s [pod6]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" \
      && mark "staged+pushed $s" || mark "PUSH-DEFERRED gen $s"
    step=$((10#$s))
    roll_one results/phrase_artifacts/dev9q_g_v9adv_$s.parquet results/analysis/v9_adv_curve.jsonl v9_adv $step
    roll_one results/phrase_artifacts/dev9q_g_v9pol_$s.parquet results/analysis/v9_pol_curve.jsonl v9_pol $step
  else
    sleep 600
  fi
done
