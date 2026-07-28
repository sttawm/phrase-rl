#!/usr/bin/env bash
# Gen-only half of the split dev8 pipeline: generate probe8 phrases for checkpoints
# on a pod that has .venv-gen + adapters, stage them in results/phrase_artifacts/
# (tracked; data/ is gitignored) and push — a roll-only pod picks them up via git.
#   STEPS="0280 0300 0320 0340" GTAG=dev8late bash scripts/v7_dev8_gen_only.sh
#   PROBE_CONTEXTS/PROBE_TRACES env select the input condition (default nominal probe8).
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export PROBE_CONTEXTS=${PROBE_CONTEXTS:-results/phrase_artifacts/contexts_probe8.parquet}
export PROBE_TRACES=${PROBE_TRACES:-results/phrase_artifacts/traces_probe8.parquet}
cd /workspace/phrase-rl
GEN=/workspace/phrase-rl/.venv-gen/bin/python
TAG=${GTAG:-dev8}
mark() { echo "[genonly $(date -u +%H:%M)] $*" | tee -a /workspace/dev8gen.log; }
# SKIP_WAIT=1: run alongside a live roll (Qwen ~20G + sim ~7G fits 46G; the only
# tight window is overlap with the resident runner's own gen phase, ~40G, still ok)
wait_idle() { [ "${SKIP_WAIT:-0}" = 1 ] && return 0; while pgrep -f "[p]hase0c_rollout" >/dev/null; do sleep 240; done; }

for s in ${STEPS:?set STEPS}; do
  g=results/phrase_artifacts/dev8q_g_${TAG}_$s.parquet
  [ -f "$g" ] && { mark "skip $s (staged)"; continue; }
  d=/workspace/v7_adapters/step_$s
  [ -f "$d/adapter_model.safetensors" ] || { mark "no adapter $s"; continue; }
  wait_idle
  mark "gen-only $s ($TAG)"
  $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/dev8_gen_$s.log 2>&1 || { mark "GEN FAIL $s"; continue; }
  cp data/rval_greedy.parquet "$g"
  cp data/rval_sampled.parquet results/phrase_artifacts/dev8q_s_${TAG}_$s.parquet
  mark "staged $s"
done
timeout 300 bash -c "git add results/phrase_artifacts/dev8q_*.parquet && git commit -q -m 'dev8 split-pipeline: staged $TAG phrases [pod]' && git pull -q --rebase && git push -q" \
  && mark "PUSHED staged phrases" || mark "PUSH-DEFERRED staged phrases"
mark "GEN-ONLY-COMPLETE ($TAG)"
