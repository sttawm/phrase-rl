#!/usr/bin/env bash
# v7f REPAIR eval, gen side (runs on pod5: has .venv-gen + git archive access).
# Every poll: pull, find newest v7f archived checkpoint >= last-staged + STRIDE,
# extract its adapter, generate adversarial-tagged rewrites (probe8_adv), stage
# to results/phrase_artifacts/dev8q_{g,s}_v7fadv_<step>.parquet, push. A roller
# (scripts/v7f_adv_roll_loop.sh) rolls staged steps as they appear.
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8_adv.parquet
export PROBE_TRACES=results/phrase_artifacts/traces_probe8_adv.parquet
export GEN_TAG="[input: adversarially reworded]"
cd /workspace/phrase-rl
GEN=/workspace/phrase-rl/.venv-gen/bin/python
STRIDE=${STRIDE:-50}
mark() { echo "[v7fadv-gen $(date -u +%H:%M)] $*" | tee -a /workspace/v7fadv.log; }

while true; do
  timeout 120 git pull -q 2>/dev/null
  last=$(ls results/phrase_artifacts/dev8q_g_v7fadv_*.parquet 2>/dev/null | sed 's/.*_0*\([0-9]*\)\.parquet/\1/' | sort -n | tail -1)
  last=${last:-0}
  # next ARCHIVED step >= last+STRIDE (checkpoints save every 10; stride rounds
  # up to whatever actually exists, so no target can wedge the loop)
  next=$(ls results/checkpoints/archive/ 2>/dev/null | grep -oE "v7f_step_[0-9]+" | grep -oE "[0-9]+$" | sort -n | awk -v t=$((last + STRIDE)) '$1 >= t' | head -1)
  if [ -n "$next" ]; then
    s=$(printf "%04d" $((10#$next)))
    d=/workspace/v7f_adapters/step_$s
    if [ ! -f "$d/adapter_model.safetensors" ]; then
      mkdir -p /tmp/v7fck && rm -rf /tmp/v7fck/*
      cat results/checkpoints/archive/v7f_step_$s.tgz.part-* 2>/dev/null | tar xzf - -C /tmp/v7fck 2>/dev/null
      src=$(find /tmp/v7fck -name adapter_model.safetensors | head -1)
      [ -n "$src" ] && { mkdir -p "$d"; cp "$(dirname $src)"/* "$d"/; } || { mark "no archive for $s yet"; sleep 600; continue; }
    fi
    mark "gen adv $s"
    $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/v7fadv_gen_$s.log 2>&1 || { mark "GEN FAIL $s"; sleep 600; continue; }
    cp data/rval_greedy.parquet results/phrase_artifacts/dev8q_g_v7fadv_$s.parquet
    cp data/rval_sampled.parquet results/phrase_artifacts/dev8q_s_v7fadv_$s.parquet
    mark "gen polish $s"
    GEN_TAG="[input: original wording]" \
      PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8.parquet \
      PROBE_TRACES=results/phrase_artifacts/traces_probe8.parquet \
      $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/v7fpol_gen_$s.log 2>&1 || { mark "POL GEN FAIL $s"; }
    cp data/rval_greedy.parquet results/phrase_artifacts/dev8q_g_v7fpol_$s.parquet 2>/dev/null
    cp data/rval_sampled.parquet results/phrase_artifacts/dev8q_s_v7fpol_$s.parquet 2>/dev/null
    timeout 300 bash -c "git add results/phrase_artifacts/dev8q_*v7fadv*.parquet results/phrase_artifacts/dev8q_*v7fpol*.parquet && git commit -q -m 'v7f eval phrases $s [pod]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" \
      && mark "staged+pushed $s" || mark "PUSH-DEFERRED $s"
  else
    sleep 900
  fi
done
