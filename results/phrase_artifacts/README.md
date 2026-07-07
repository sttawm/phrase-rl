# Generated phrase artifacts (durable backup)

Expensive-to-regenerate Gemini/Qwen outputs (text only, no images). Backed up
here because they cost ~$15 + hours of API/GPU time; the image-bearing context
parquets in `data/` are gitignored (regenerate deterministically via
`extract_contexts.py` / `phase0c_match_contexts.py`).

- `rephrases_val_0b.parquet` — Gemini 3.1 Pro, 250 val ctx × 32 + traces (0b)
- `rephrases35flash_val_0b.parquet` — Gemini 3.5 Flash, 250 × 32 + traces (0b)
- `qwen_rephrases_val_0b.parquet` — Qwen3.5-9B base, 250 × 32 (0b)
- `phrases_0c.parquet` — 4 SIMPLER task instructions × ~32 (0c rollouts + scoring)
- `teacher_train.parquet` — Gemini 3.1 Pro, 2000 train ctx × 16 + traces (Phase 1 SFT teacher)
