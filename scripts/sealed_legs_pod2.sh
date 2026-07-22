#!/usr/bin/env bash
# Sealed legs, RULES-FREE order (rules arms deferred to rules-v3 post-certification)
set -uo pipefail
cd /workspace/phrase-rl
until ls results/sealed/ph_sealed_sftv2.parquet >/dev/null 2>&1; do git pull -q --rebase 2>/dev/null; sleep 60; done
cp results/sealed/ph_sealed_*.parquet data/ 2>/dev/null || true
for f in data/ph_sealed_*.parquet; do :; done
bash scripts/run_sealed_leg.sh sft_v2 data/ph_sealed_sftv2.parquet 12
bash scripts/run_sealed_leg.sh anchors data/ph_sealed_anchors.parquet 12
bash scripts/run_sealed_leg.sh frozen_selftrace data/ph_sealed_frozen_selftrace.parquet 12
bash scripts/run_sealed_leg.sh frozen_gemini data/ph_sealed_frozen_gemini.parquet 12
bash scripts/run_sealed_leg.sh v6_rl data/ph_sealed_v6.parquet 12
bash scripts/run_sealed_leg.sh oracle_pool data/ph_sealed_oraclepool.parquet 1
echo "POD2-SEALED-BATCH-DONE" | tee -a /workspace/sealed.log
