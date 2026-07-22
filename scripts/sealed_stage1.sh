#!/usr/bin/env bash
# Sealed stage 1: assets + rules arm (the frozen plan's staged firing)
set -uo pipefail
export FINAL_EVAL=1
cd /workspace/phrase-rl
mark() { echo "[sealed $(date +%H:%M:%S)] $*" | tee -a /workspace/sealed.log; }
mark "STAGE1 START: assets"
cd /workspace/INT-ACT && /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/scripts/build_sealed_assets.py --stage render --frames-out /workspace/phrase-rl/data/sealed_frames.parquet 2>&1 | tail -2 | tee -a /workspace/sealed.log || { mark "RENDER FAILED"; exit 1; }
cd /workspace/phrase-rl
eval "$(grep GEMINI_API_KEY ~/.bashrc)"
PYTHONPATH=src .venv-gen/bin/python scripts/build_sealed_assets.py --stage gemini 2>&1 | tee -a /workspace/sealed.log | tail -3 || { mark "GEMINI FAILED"; exit 1; }
PYTHONPATH=src .venv-gen/bin/python scripts/build_sealed_assets.py --stage selftrace 2>&1 | tail -3 | tee -a /workspace/sealed.log || { mark "SELFTRACE FAILED"; exit 1; }
PYTHONPATH=src .venv-gen/bin/python -m phrase_rl.sft17k_rules_gen --assets data/sealed_assets_selftrace.parquet --arm rules_qwen --out data/phrases_rules_sealed.parquet 2>&1 | grep "\[rules" | tee -a /workspace/sealed.log || { mark "RULES GEN FAILED"; exit 1; }
mkdir -p results/sealed && cp data/sealed_assets_gemini.parquet data/sealed_assets_selftrace.parquet data/phrases_rules_sealed.parquet results/sealed/ && git add results/sealed && git commit -q -m "sealed assets + rules phrases [pod]" && git pull -q --rebase && git push -q
mark "assets committed; pausing search for the rules leg"
tmux kill-session -t search1 2>/dev/null; pkill -f "[p]hase0c_rollout"; sleep 10
bash scripts/run_sealed_leg.sh rules_qwen data/phrases_rules_sealed.parquet 12
mark "rules leg done; resuming search"
tmux new-session -d -s search1 "TASK_FILTER=\"spoon|carrotplate|stack\" bash /workspace/phrase-rl/scripts/search_worker.sh"
mark "STAGE1 COMPLETE"
