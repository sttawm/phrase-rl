#!/bin/bash
# Backfill dev10q phrase files for the mid-curve checkpoints (100-180) from
# the archives evacuated to this pod. Mirrors the gen worker's extraction +
# generation flow exactly; pushes each step's pair with the retry pattern.
cd /workspace/phrase-rl
GEN=/workspace/phrase-rl/.venv-gen/bin/python
mark() { echo "[midgen $(date +%H:%M)] $*" >> /workspace/midgen.log; }

for n in 100 110 120 130 140 150 160 170 180; do
  s=$(printf "%04d" $n)
  [ -f "results/phrase_artifacts/dev10q_g_v10adv_$s.parquet" ] && { mark "skip $s (exists)"; continue; }
  d=/workspace/v10_adapters/step_$s
  if [ ! -f "$d/adapter_model.safetensors" ]; then
    mkdir -p /tmp/v10ck && rm -rf /tmp/v10ck/*
    cat results/checkpoints/archive/v10_step_$s.tgz.part-* 2>/dev/null | tar xzf - -C /tmp/v10ck 2>/dev/null
    src=$(find /tmp/v10ck -name adapter_model.safetensors | head -1)
    [ -n "$src" ] && { mkdir -p "$d"; cp "$(dirname $src)"/* "$d"/; } || { mark "NO ARCHIVE $s"; continue; }
  fi
  mark "gen adv $s"
  GEN_TAG="" PROMPT_FAMILY=bare \
    PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8_adv.parquet \
    PROBE_TRACES=results/phrase_artifacts/traces_probe8_adv.parquet \
    $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/midgen_adv_$s.log 2>&1 || { mark "GEN FAIL adv $s"; continue; }
  cp data/rval_greedy.parquet results/phrase_artifacts/dev10q_g_v10adv_$s.parquet
  mark "gen pol $s"
  GEN_TAG="" PROMPT_FAMILY=bare \
    PROBE_CONTEXTS=results/phrase_artifacts/contexts_probe8.parquet \
    PROBE_TRACES=results/phrase_artifacts/traces_probe8.parquet \
    $GEN scripts/gen_ckpt_phrases.py "$d" 1 > /workspace/midgen_pol_$s.log 2>&1 || { mark "GEN FAIL pol $s"; continue; }
  cp data/rval_greedy.parquet results/phrase_artifacts/dev10q_g_v10pol_$s.parquet
  git add results/phrase_artifacts/dev10q_g_v10adv_$s.parquet results/phrase_artifacts/dev10q_g_v10pol_$s.parquet
  git commit -q -m "midgen backfill: v10 eval phrases $s [pod6]"
  for i in 1 2 3 4; do
    timeout 300 bash -c "git -c rebase.autoStash=true pull -q --rebase && git push -q" && { mark "pushed $s"; break; }
    git rebase --abort 2>/dev/null; sleep $((30 * i))
  done
done
mark "MIDGEN-COMPLETE"
