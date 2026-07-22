#!/usr/bin/env bash
# Generate ALL sealed arm phrases (runs alongside rollouts; GPU-share safe)
set -uo pipefail
export FINAL_EVAL=1
cd /workspace/phrase-rl
eval "$(grep GEMINI_API_KEY ~/.bashrc)"
mark() { echo "[sealedgen $(date +%H:%M:%S)] $*" | tee -a /workspace/sealedgen.log; }
G() { PYTHONPATH=src .venv-gen/bin/python "$@" >> /workspace/sealedgen.log 2>&1; }
cp results/sealed/sealed_assets_gemini.parquet results/sealed/sealed_assets_selftrace.parquet data/ 2>/dev/null || true
[ -f data/sealed_frames.parquet ] || { mark "NO FRAMES"; exit 1; }
mark anchors;      G scripts/build_sealed_arm_phrases.py --mode anchors --out data/ph_sealed_anchors.parquet || exit 1
mark frozen-gem;   G scripts/build_sealed_arm_phrases.py --mode frozen --trace-source gemini --out data/ph_sealed_frozen_gemini.parquet || exit 1
mark frozen-self;  G scripts/build_sealed_arm_phrases.py --mode frozen --trace-source selftrace --out data/ph_sealed_frozen_selftrace.parquet || exit 1
mark v6;           G scripts/build_sealed_arm_phrases.py --mode v6 --out data/ph_sealed_v6.parquet || exit 1
mark sft-v2;       G -m phrase_rl.sft17k_generate_eval --adapter results/checkpoints/sft17k_v2/final --assets data/sealed_assets_gemini.parquet --trace-assets data/sealed_assets_selftrace.parquet --arm sft_v2 --out data/ph_sealed_sftv2.parquet || exit 1
mark rulesv2-self; G -m phrase_rl.sft17k_rules_gen --assets data/sealed_assets_selftrace.parquet --rules-path results/analysis/b4_phrasing_rules_v2.md --arm rules_v2_selftrace --out data/ph_sealed_rulesv2_self.parquet || exit 1
mark rulesv2-gem;  G -m phrase_rl.sft17k_rules_gen --assets data/sealed_assets_gemini.parquet --rules-path results/analysis/b4_phrasing_rules_v2.md --arm rules_v2_gemini --out data/ph_sealed_rulesv2_gem.parquet || exit 1
mark rules-gemini-exec; G scripts/build_sealed_arm_phrases.py --mode rulesgemini --rules-path results/analysis/b4_phrasing_rules_v2.md --out data/ph_sealed_rulesgemini.parquet || exit 1
mark oracle-pool;  G scripts/build_sealed_arm_phrases.py --mode oracle --out data/ph_sealed_oraclepool.parquet || exit 1
mkdir -p results/sealed && cp data/ph_sealed_*.parquet results/sealed/ && git add results/sealed && git commit -q -m "sealed: all arm phrases [pod]" && git pull -q --rebase && git push -q
mark "GEN-ALL-DONE"
