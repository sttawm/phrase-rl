#!/bin/bash
# v12 single-pod checkpoint evaluator (A27): generate prompt-B+ phrases from each
# archived v12 checkpoint and roll them on the NATURAL-input val probe (primary
# metric). One pod does both roles — no claims, no fan-out.
#   STRIDE=10 bash scripts/v12_eval_loop.sh
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
mark() { echo "[v12eval $(date -u +%H:%M)] $*" >> /workspace/v12eval.log; }

mkdir -p results/analysis/v12cells /workspace/v12_adapters

# --- baselines on the SAME probe/protocol, produced once before checkpoint cells:
#   nat_0000  = frozen base Qwen + prompt B+ (RL step 0)
#   nat_pass  = the natural inputs rolled with NO rephraser (passthrough)
roll_and_cell() { # $1 phrases parquet (repo-relative), $2 cell name, $3 max episode id (default 23; judges use 95 -> 4x episodes, same CRN prefix)
  local MAXEP=${3:-23}
  rm -f data/dev12_g_out.parquet
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases /workspace/phrase-rl/$1 --episode-ids $(seq 0 $MAXEP) --repeats 1 \
    --out /workspace/phrase-rl/data/dev12_g_out.parquet > /workspace/v12roll_$2.log 2>&1
  rc=$?; cd /workspace/phrase-rl
  [ $rc != 0 ] && { mark "ROLL FAIL $2"; return 1; }
  STEP=$2 CELLF=results/analysis/v12cells/$2.json $VLA - <<'PYEOF'
import json, os
import pandas as pd
g = pd.read_parquet("data/dev12_g_out.parquet")
rec = {"step": os.environ["STEP"], "probe": "v12_nat",
       "n": int(len(g)), "pooled": round(float(g.success.mean() * 100), 2),
       "per_task": {t: round(float(x.success.mean() * 100), 1) for t, x in g.groupby("task")}}
json.dump(rec, open(os.environ["CELLF"], "w"))
print(os.environ["STEP"], "pooled", rec["pooled"])
PYEOF
  for i in 1 2 3; do
    timeout 300 bash -c "git add results/analysis/v12cells/$2.json && git commit -q -m 'v12 baseline cell $2 [e4]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" && { mark "DONE $2"; return 0; }
    git rebase --abort 2>/dev/null; sleep 30
  done
}

if [ ! -f results/analysis/v12cells/nat24_pass.json ]; then
  mark "baseline: passthrough (no rephraser)"
  $GEN - <<'PYEOF'
import pandas as pd
c = pd.read_parquet("results/phrase_artifacts/contexts_probe8_nat24.parquet")
pd.DataFrame({"task": c.task, "arm": "passthrough", "phrase": c.instruction,
              "instruction": c.instruction, "episode_id": c.episode_id}).to_parquet(
    "results/phrase_artifacts/dev12q_g_nat24_pass.parquet", index=False)
PYEOF
  roll_and_cell results/phrase_artifacts/dev12q_g_nat24_pass.parquet nat24_pass
fi

if [ ! -f results/analysis/v12cells/nat24_0000.json ]; then
  mark "baseline: BASE qwen + prompt B+ (step 0)"
  GEN_TAG="" PROMPT_FAMILY=bplusimg \
    PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8_nat24.parquet \
    PROBE_TRACES=results/phrase_artifacts/traces_probe8.parquet \
    $GEN scripts/gen_ckpt_phrases.py BASE 1 > /workspace/v12gen_base.log 2>&1 \
    && cp data/rval_greedy.parquet results/phrase_artifacts/dev12q_g_nat24_0000.parquet \
    && roll_and_cell results/phrase_artifacts/dev12q_g_nat24_0000.parquet nat24_0000 \
    || mark "BASE GEN FAIL"
fi

# v12 spec: paired 768-episode judge at STEP 0 (episode-ids 0..95 -- superset
# of the stride cells' 0..23, so decode seeds crc32(task|ep|rep) are shared and
# the step-150 judge on the same range is a PAIRED comparison). Success is
# measured against THIS cell, not the inherited 42.19.
if [ ! -f results/analysis/v12cells/nat24_judge_0000.json ] \
   && [ -f results/phrase_artifacts/dev12q_g_nat24_0000.parquet ]; then
  mark "JUDGE step 0 (768 episodes: rows x4 at episode offsets +24/+48/+72)"
  $GEN - <<'PYX4'
import pandas as pd
d = pd.read_parquet("results/phrase_artifacts/dev12q_g_nat24_0000.parquet")
out = pd.concat([d.assign(episode_id=d.episode_id + k) for k in (0, 24, 48, 72)],
                ignore_index=True)
out.to_parquet("results/phrase_artifacts/dev12q_g_nat24_0000_x4.parquet", index=False)
print("judge parquet:", len(out), "rows")
PYX4
  roll_and_cell results/phrase_artifacts/dev12q_g_nat24_0000_x4.parquet nat24_judge_0000 95
fi

while true; do
  timeout 240 git -c rebase.autoStash=true pull -q --rebase 2>/dev/null \
    || { git rebase --abort 2>/dev/null; git reset --hard -q origin/main; }

  # step-150 judge: when the checkpoint lands, it outranks stride cells.
  if [ -f "results/checkpoints/archive/v12_step_0150.manifest.json" ] \
     && [ ! -f results/analysis/v12cells/nat24_judge_0150.json ] \
     && [ ! -f results/analysis/v12claims/judge150.claim ]; then
    echo "${POD:-e4}" > results/analysis/v12claims/judge150.claim
    git add results/analysis/v12claims/judge150.claim \
      && git commit -q -m "claim v12 judge150 [${POD:-e4}]" \
      && timeout 200 git -c rebase.autoStash=true pull -q --rebase && timeout 200 git push -q || true
    d=/workspace/v12_adapters/step_0150
    if [ ! -f "$d/adapter_model.safetensors" ]; then
      mkdir -p /tmp/v12ck && rm -rf /tmp/v12ck/*
      cat results/checkpoints/archive/v12_step_0150.tgz.part-* | tar xzf - -C /tmp/v12ck
      src=$(find /tmp/v12ck -name adapter_model.safetensors | head -1)
      [ -n "$src" ] && { mkdir -p "$d"; cp "$(dirname $src)"/* "$d"/; }
    fi
    if [ -f "$d/adapter_model.safetensors" ]; then
      mark "gen judge150 phrases"
      GEN_TAG="" PROMPT_FAMILY=bplusimg \
        PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8_nat24.parquet \
        PROBE_TRACES=results/phrase_artifacts/traces_probe8.parquet \
        $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/v12gen_j150.log 2>&1 \
        && cp data/rval_greedy.parquet results/phrase_artifacts/dev12q_g_nat24_j150.parquet \
        || { mark "JUDGE150 GEN FAIL"; }
      if [ -f results/phrase_artifacts/dev12q_g_nat24_j150.parquet ]; then
        mark "JUDGE step 150 (768 episodes: rows x4, same construction as step 0)"
        $GEN - <<'PYX4B'
import pandas as pd
d = pd.read_parquet("results/phrase_artifacts/dev12q_g_nat24_j150.parquet")
out = pd.concat([d.assign(episode_id=d.episode_id + k) for k in (0, 24, 48, 72)],
                ignore_index=True)
out.to_parquet("results/phrase_artifacts/dev12q_g_nat24_j150_x4.parquet", index=False)
print("judge parquet:", len(out), "rows")
PYX4B
        roll_and_cell results/phrase_artifacts/dev12q_g_nat24_j150_x4.parquet nat24_judge_0150 95
      fi
    fi
  fi

  # OLDEST archived stride-multiple step that has neither a cell nor a live claim.
  # Oldest-first backfills the pre-nat24 checkpoints so the whole curve is on one
  # metric; the claim files let several pods share the queue without duplicating.
  mkdir -p results/analysis/v12claims
  next=""
  for cand in $(ls results/checkpoints/archive/ 2>/dev/null | grep -oE "v12_step_[0-9]+" \
                | grep -oE "[0-9]+$" | sort -n -u | awk -v s=$STRIDE '$1 % s == 0'); do
    cs=$(printf "%04d" $((10#$cand)))
    [ -f "results/analysis/v12cells/nat24_$cs.json" ] && continue
    [ -f "results/analysis/v12claims/$cs.claim" ] && continue
    echo "${POD:-e4}" > "results/analysis/v12claims/$cs.claim"
    if git add "results/analysis/v12claims/$cs.claim" \
       && git commit -q -m "claim v12 nat24 cell $cs [${POD:-e4}]" \
       && timeout 200 git -c rebase.autoStash=true pull -q --rebase \
       && timeout 200 git push -q \
       && grep -q "${POD:-e4}" "results/analysis/v12claims/$cs.claim" 2>/dev/null; then
      next=$cand; break
    fi
    git rebase --abort 2>/dev/null; git reset --hard -q origin/main
  done
  if [ -z "$next" ]; then mark "idle (nothing unclaimed)"; sleep 600; continue; fi
  s=$(printf "%04d" $((10#$next)))

  d=/workspace/v12_adapters/step_$s
  if [ ! -f "$d/adapter_model.safetensors" ]; then
    mkdir -p /tmp/v12ck && rm -rf /tmp/v12ck/*
    cat results/checkpoints/archive/v12_step_$s.tgz.part-* 2>/dev/null | tar xzf - -C /tmp/v12ck 2>/dev/null
    src=$(find /tmp/v12ck -name adapter_model.safetensors | head -1)
    [ -n "$src" ] && { mkdir -p "$d"; cp "$(dirname $src)"/* "$d"/; } \
      || { mark "archive incomplete for $s"; sleep 600; continue; }
  fi

  mark "gen nat $s (prompt B+)"
  GEN_TAG="" PROMPT_FAMILY=bplusimg \
    PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8_nat24.parquet \
    PROBE_TRACES=results/phrase_artifacts/traces_probe8.parquet \
    $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/v12gen_$s.log 2>&1 \
    || { mark "GEN FAIL $s"; sleep 300; continue; }
  cp data/rval_greedy.parquet results/phrase_artifacts/dev12q_g_nat24_$s.parquet

  mark "roll nat $s greedy(192)"
  rm -f data/dev12_g_out.parquet
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases /workspace/phrase-rl/results/phrase_artifacts/dev12q_g_nat24_$s.parquet \
    --episode-ids $(seq 0 23) --repeats 1 \
    --out /workspace/phrase-rl/data/dev12_g_out.parquet > /workspace/v12roll_$s.log 2>&1
  rc=$?
  cd /workspace/phrase-rl
  [ $rc != 0 ] && { mark "ROLL FAIL $s"; sleep 300; continue; }

  STEP=$s CELLF=results/analysis/v12cells/nat24_$s.json $VLA - <<'PYEOF' || { mark "MERGE FAIL $s"; continue; }
import json, os
import pandas as pd
g = pd.read_parquet("data/dev12_g_out.parquet")
rec = {"step": int(os.environ["STEP"]), "probe": "v12_nat",
       "n": int(len(g)), "pooled": round(float(g.success.mean() * 100), 2),
       "per_task": {t: round(float(x.success.mean() * 100), 1) for t, x in g.groupby("task")}}
with open(os.environ["CELLF"], "w") as f:
    json.dump(rec, f)
print("v12_nat step", rec["step"], "pooled", rec["pooled"])
PYEOF

  for i in 1 2 3; do
    timeout 300 bash -c "git add results/analysis/v12cells/nat24_$s.json results/phrase_artifacts/dev12q_g_nat24_$s.parquet && git commit -q -m "v12 nat24 cell $s [${POD:-e4}]" && git -c rebase.autoStash=true pull -q --rebase && git push -q" \
      && { mark "DONE nat $s"; break; }
    git rebase --abort 2>/dev/null; sleep $((30 * i))
  done
done
