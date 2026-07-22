#!/usr/bin/env bash
# Pod2 sealed legs: waits for phrases + own purification drain, rolls arms in priority order
set -uo pipefail
cd /workspace/phrase-rl
mark() { echo "[sealed2 $(date +%H:%M:%S)] $*" | tee -a /workspace/sealed.log; }
until git pull -q --rebase 2>/dev/null && [ -f results/sealed/ph_sealed_rulesv2_self.parquet ]; do sleep 120; done
until [ -f results/search/ramekin_pure1_results.json ] && [ -f results/search/spoon_pure1_results.json ] && [ -f results/search/stack_pure1_results.json ]; do git pull -q --rebase 2>/dev/null; sleep 120; done
tmux kill-session -t search2 2>/dev/null; pkill -f "[p]hase0c_rollout"; sleep 8
cp results/sealed/ph_sealed_*.parquet data/ 2>/dev/null
bash scripts/run_sealed_leg.sh rules_v2_selftrace data/ph_sealed_rulesv2_self.parquet 12
bash scripts/run_sealed_leg.sh rules_v2_gemini data/ph_sealed_rulesv2_gem.parquet 12
bash scripts/run_sealed_leg.sh sft_v2 data/ph_sealed_sftv2.parquet 12
bash scripts/run_sealed_leg.sh frozen_selftrace data/ph_sealed_frozen_selftrace.parquet 12
bash scripts/run_sealed_leg.sh frozen_gemini data/ph_sealed_frozen_gemini.parquet 12
mark "POD2-LEGS-BATCH1-DONE"
