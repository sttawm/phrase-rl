#!/usr/bin/env bash
# Corpus grip-screen sweep, split design (user 2026-07-23):
#   leg A: all 2000 episodes x F=8  -> pooled corpus tables (grip_sweep_full)
#   leg B: 100-episode subset x F=16 -> resolvable per-scene partial orders
# Gemini cover35 traces ground add_color/family edits. Suggestive-tier output;
# feeds nominations to rollout certification (EXPERIMENT.md epistemic status).
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
git pull --no-edit -q || true
export PYTHONPATH=/workspace/phrase-rl/src:${PYTHONPATH:-}
PY=${SCREEN_PY:-/workspace/INT-ACT/.venv/bin/python}
$PY -c "from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy" 2>/dev/null \
  || PY=/root/venv/bin/python

$PY -m phrase_rl.phrase_grip_screen \
  --contexts data/contexts_train_multit16.parquet \
  --stats-contexts results/phrase_artifacts/chunk_stats.parquet \
  --episodes 2000 --frames 8 --k 4 --seed 0 \
  --out results/analysis/grip_sweep_full.json 2>&1 | tee /workspace/sweepA.log

git add results/analysis/grip_sweep_full.json && git commit -q -m "grip sweep leg A: 2000eps F=8 [pod]" && git pull -q --rebase && git push -q || true

$PY -m phrase_rl.phrase_grip_screen \
  --contexts data/contexts_train_multit16.parquet \
  --stats-contexts results/phrase_artifacts/chunk_stats.parquet \
  --episodes 100 --frames 16 --k 4 --seed 7 \
  --out results/analysis/grip_sweep_hires100.json 2>&1 | tee /workspace/sweepB.log

git add results/analysis/grip_sweep_hires100.json && git commit -q -m "grip sweep leg B: 100eps F=16 [pod]" && git pull -q --rebase && git push -q || true
echo GRIP-SWEEP-DONE
