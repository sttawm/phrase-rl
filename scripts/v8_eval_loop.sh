#!/usr/bin/env bash
# v8 eval, gen side (pod6): stride-10 checkpoint watcher, TAG-FREE generation for
# BOTH conditions (v8 never saw tags). Stages dev8q_{g,s}_v8{adv,pol}_<step>.parquet.
# Roller: scripts/eval_roll_loop.sh picks staged steps up as they appear.
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
GEN=/workspace/phrase-rl/.venv-gen/bin/python
STRIDE=${STRIDE:-10}
mark() { echo "[v8-gen $(date -u +%H:%M)] $*" | tee -a /workspace/v8gen.log; }

while true; do
  timeout 120 git pull -q 2>/dev/null
  last=$(ls results/phrase_artifacts/dev8q_g_v8adv_*.parquet 2>/dev/null | sed 's/.*_0*\([0-9]*\)\.parquet/\1/' | sort -n | tail -1)
  last=${last:-0}
  next=$(ls results/checkpoints/archive/ 2>/dev/null | grep -oE "v8_step_[0-9]+" | grep -oE "[0-9]+$" | sort -n | awk -v t=$((last + STRIDE)) '$1 >= t' | head -1)
  if [ -n "$next" ]; then
    s=$(printf "%04d" $((10#$next)))
    d=/workspace/v8_adapters/step_$s
    if [ ! -f "$d/adapter_model.safetensors" ]; then
      mkdir -p /tmp/v8ck && rm -rf /tmp/v8ck/*
      cat results/checkpoints/archive/v8_step_$s.tgz.part-* 2>/dev/null | tar xzf - -C /tmp/v8ck 2>/dev/null
      src=$(find /tmp/v8ck -name adapter_model.safetensors | head -1)
      [ -n "$src" ] && { mkdir -p "$d"; cp "$(dirname $src)"/* "$d"/; } || { mark "no archive for $s yet"; sleep 600; continue; }
    fi
    mark "gen adv $s (tag-free)"
    GEN_TAG="" \
      PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8_adv.parquet \
      PROBE_TRACES=results/phrase_artifacts/traces_probe8_adv.parquet \
      $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/v8adv_gen_$s.log 2>&1 || { mark "GEN FAIL $s"; sleep 600; continue; }
    cp data/rval_greedy.parquet results/phrase_artifacts/dev8q_g_v8adv_$s.parquet
    cp data/rval_sampled.parquet results/phrase_artifacts/dev8q_s_v8adv_$s.parquet
    mark "gen polish $s (tag-free)"
    GEN_TAG="" \
      PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8.parquet \
      PROBE_TRACES=results/phrase_artifacts/traces_probe8.parquet \
      $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/v8pol_gen_$s.log 2>&1 || { mark "POL GEN FAIL $s"; }
    cp data/rval_greedy.parquet results/phrase_artifacts/dev8q_g_v8pol_$s.parquet 2>/dev/null
    cp data/rval_sampled.parquet results/phrase_artifacts/dev8q_s_v8pol_$s.parquet 2>/dev/null
    timeout 300 bash -c "git add results/phrase_artifacts/dev8q_*v8adv*.parquet results/phrase_artifacts/dev8q_*v8pol*.parquet && git commit -q -m 'v8 eval phrases $s [pod]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" \
      && mark "staged+pushed $s" || mark "PUSH-DEFERRED $s"
  else
    sleep 600
  fi
done
