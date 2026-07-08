#!/usr/bin/env bash
# Regenerate ALL teacher traces + 16-lists in CoVer format with gemini-3.5-flash.
# Robustness: gemini_cover_rephrase preflights, saves incrementally every 25,
# hard-stops on depleted credits, and resumes from the output parquet — re-run
# this script after any failure and it continues where it left off.
set -euxo pipefail
export UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
cd /workspace/phrase-rl

.venv/bin/python -m phrase_rl.gemini_cover_rephrase \
  --contexts data/contexts_train.parquet --out data/cover35_train_raw.parquet \
  --n 16 --model gemini-3.5-flash --concurrency 6

.venv/bin/python -m phrase_rl.gemini_cover_rephrase \
  --contexts data/contexts_val_0b.parquet --out data/cover35_val_raw.parquet \
  --n 16 --model gemini-3.5-flash --concurrency 6

# Build trainer-facing parquets: episode_index, t, instruction, trace, rephrases.
# Trace = reasoning before 'Reworded Instructions:'; rephrases kept for the
# source-aug pool / SFT ablation / comparison — NEVER as training candidates
# (candidates are always Qwen's own generations).
.venv/bin/python - <<'PY'
import pandas as pd, sys
sys.path.insert(0, "src")
from phrase_rl.cover_prompt import extract_trace
for split in ["train", "val"]:
    df = pd.read_parquet(f"data/cover35_{split}_raw.parquet")
    df["trace"] = df["raw_response"].map(lambda r: extract_trace(str(r)))
    n_bad = int(df["trace"].isna().sum())
    df = df.dropna(subset=["trace"])
    out = df[["episode_index", "t", "instruction", "trace", "rephrases"]]
    out.to_parquet(f"results/phrase_artifacts/cover35_teacher_{split}.parquet", index=False)
    print(f"{split}: {len(out)} contexts ({n_bad} unusable responses dropped)")
PY

git add results/phrase_artifacts
git -c user.name=pod -c user.email=pod@runpod commit -m "CoVer-format 3.5-flash teacher traces + 16-lists (train+val) [pod]" || true
git -c user.name=pod -c user.email=pod@runpod pull --rebase --no-edit || true
git push || echo "PUSH FAILED — rerun publish manually"
echo "TRACE REGEN DONE"
