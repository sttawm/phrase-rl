#!/usr/bin/env bash
# v10 all-in-one eval worker (pod6) — PROMPT B (bare family) generation: per new archived checkpoint, sequentially
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
STRIDE=${STRIDE:-14}
mark() { echo "[v10work $(date -u +%H:%M)] $*" | tee -a /workspace/v9work.log; }

roll_one() { # $1 staged parquet, $2 OUT jsonl, $3 COND label, $4 step
  grep -q "\"step\": $4," "$2" 2>/dev/null && return 0
  mark "roll $3 $4 greedy(192)"
  rm -f data/dev9_g_out.parquet
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases /workspace/phrase-rl/$1 --episode-ids $(seq 0 23) --repeats ${REPEATS:-2} \
    --out /workspace/phrase-rl/data/dev9_g_out.parquet > /workspace/v10roll_$3_$4.log 2>&1
  rc=$?
  cd /workspace/phrase-rl
  [ $rc != 0 ] && { mark "ROLL FAIL $3 $4"; return 1; }
  # cell file = one unique path per (cond, step): concurrent consumers can
  # never collide on it, unlike the shared curve jsonl (which wedged the
  # fleet on 08-06: N pods appending -> rebase conflicts -> push dead)
  mkdir -p results/analysis/v10cells
  STEP=$4 COND=$3 OUTF=$2 CELLF=results/analysis/v10cells/${3}_${4}.json \
      $VLA - <<'PYEOF' || { mark "MERGE FAIL $3 $4"; return 1; }
import json, os
import pandas as pd
g = pd.read_parquet("data/dev9_g_out.parquet")
rec = {"step": int(os.environ["STEP"]), "probe": os.environ["COND"],
       "n": int(len(g)), "pooled": round(float(g.success.mean()*100), 2),
       "per_task": {t: round(float(x.success.mean()*100), 1) for t, x in g.groupby("task")}}
with open(os.environ["CELLF"], "w") as f:
    json.dump(rec, f)
with open(os.environ["OUTF"], "a") as f:  # local convenience copy only, never pushed
    f.write(json.dumps(rec) + "\n")
print(os.environ["COND"], "step", rec["step"], "greedy", rec["pooled"])
PYEOF
  for i in 1 2 3; do
    timeout 300 bash -c "git add results/analysis/v10cells/${3}_${4}.json && git commit -q -m 'v10 cell $3 $4 [${POD:-c}]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" && { mark "DONE $3 $4"; return 0; }
    git rebase --abort 2>/dev/null; sleep $((30 * i))
  done
  mark "PUSH-DEFERRED $3 $4"
}

claim() { # $1 tag  -> 0 iff OUR claim commit reached origin (push-arbitrated)
  local f=results/analysis/v10claims/$1.claim
  mkdir -p results/analysis/v10claims
  [ -f "$f" ] && return 1
  echo "${POD:-consumer}" > "$f"
  if timeout 200 bash -c "git add $f && git commit -q -m 'claim $1 [${POD:-c}]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" 2>/dev/null \
     && grep -q "${POD:-consumer}" "$f" 2>/dev/null; then
    return 0
  fi
  # any failure (incl. lost race -> rebase conflict): abort + hard reset so the
  # repo can NEVER be left wedged; the claim simply goes to whoever pushed first
  git rebase --abort 2>/dev/null; git reset --hard origin/main -q 2>/dev/null
  return 1
}

if [ -n "${ROLL_ONLY:-}" ]; then
  while true; do
    timeout 120 git -c rebase.autoStash=true pull -q 2>/dev/null
    did=0
    for f in $(ls results/phrase_artifacts/dev10q_g_v10adv_*.parquet results/phrase_artifacts/dev10q_g_v10pol_*.parquet results/phrase_artifacts/dev10q_g_v10ref_*.parquet 2>/dev/null | shuf); do
      [ -f "$f" ] || continue
      s=$(echo "$f" | sed 's/.*_0*\([0-9][0-9]*\)\.parquet/\1/')
      case "$f" in *v10ref*) cond=v10_ref; out=results/analysis/v10_ref_curve.jsonl;; *v10adv*) cond=v10_adv; out=results/analysis/v10_adv_curve.jsonl;; *) cond=v10_pol; out=results/analysis/v10_pol_curve.jsonl;; esac
      [ -f "results/analysis/v10cells/${cond}_${s}.json" ] && continue
      claim "${cond}_$s" || continue
      mark "consumer roll $cond $s"
      roll_one "$f" "$out" "$cond" "$s" && did=1
    done
    [ $did = 0 ] && sleep 600
  done
fi

while true; do
  timeout 120 git -c rebase.autoStash=true pull -q 2>/dev/null
  last=$(ls results/phrase_artifacts/dev10q_g_v10adv_*.parquet 2>/dev/null | sed 's/.*_0*\([0-9]*\)\.parquet/\1/' | sort -n | tail -1)
  last=${last:-0}
  # eligibility = next unrolled STRIDE-multiple (absolute), NOT last+STRIDE:
  # with last=10 (stride-10 era) the old test demanded >=17 and silently
  # skipped the archived step 14 (caught 2026-08-03)
  next=$(ls results/checkpoints/archive/ 2>/dev/null | grep -oE "v10_step_[0-9]+" | grep -oE "[0-9]+$" | sort -n | awk -v t=$last -v s=$STRIDE '$1 > t && $1 % s == 0' | head -1)
  if [ -n "$next" ]; then
    s=$(printf "%04d" $((10#$next)))
    d=/workspace/v10_adapters/step_$s
    if [ ! -f "$d/adapter_model.safetensors" ]; then
      mkdir -p /tmp/v10ck && rm -rf /tmp/v10ck/*
      cat results/checkpoints/archive/v10_step_$s.tgz.part-* 2>/dev/null | tar xzf - -C /tmp/v10ck 2>/dev/null
      src=$(find /tmp/v10ck -name adapter_model.safetensors | head -1)
      [ -n "$src" ] && { mkdir -p "$d"; cp "$(dirname $src)"/* "$d"/; } || { mark "no archive for $s yet"; sleep 600; continue; }
    fi
    mark "gen adv $s (tag-free)"
    GEN_TAG="" PROMPT_FAMILY=bare \
      PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8_adv.parquet \
      PROBE_TRACES=results/phrase_artifacts/traces_probe8_adv.parquet \
      $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/v10adv_gen_$s.log 2>&1 || { mark "GEN FAIL $s"; sleep 600; continue; }
    cp data/rval_greedy.parquet results/phrase_artifacts/dev10q_g_v10adv_$s.parquet
    cp data/rval_sampled.parquet results/phrase_artifacts/dev10q_s_v10adv_$s.parquet
    mark "gen polish $s (tag-free)"
    GEN_TAG="" PROMPT_FAMILY=bare \
      PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8.parquet \
      PROBE_TRACES=results/phrase_artifacts/traces_probe8.parquet \
      $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/v10pol_gen_$s.log 2>&1 || { mark "POL GEN FAIL $s"; }
    cp data/rval_greedy.parquet results/phrase_artifacts/dev10q_g_v10pol_$s.parquet 2>/dev/null
    cp data/rval_sampled.parquet results/phrase_artifacts/dev10q_s_v10pol_$s.parquet 2>/dev/null
    pushed=0
    for i in 1 2 3 4; do
      timeout 300 bash -c "git add results/phrase_artifacts/dev10q_*v10adv*.parquet results/phrase_artifacts/dev10q_*v10pol*.parquet && git commit -q -m 'v10 eval phrases $s [pod6]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" && { pushed=1; break; }
      git rebase --abort 2>/dev/null; sleep $((30 * i))
    done
    [ $pushed = 1 ] && mark "staged+pushed $s" || mark "PUSH-DEFERRED gen $s"
    step=$((10#$s))
    if [ -z "${SKIP_ROLL:-}" ]; then
      roll_one results/phrase_artifacts/dev10q_g_v10adv_$s.parquet results/analysis/v10_adv_curve.jsonl v10_adv $step
      roll_one results/phrase_artifacts/dev10q_g_v10pol_$s.parquet results/analysis/v10_pol_curve.jsonl v10_pol $step
    fi
  else
    sleep 600
  fi
done
