---
name: phase0c-out-parquet-resume
description: phase0c_rollout resumes/accumulates on its --out parquet — rm out files between rolls; n differing from design (192) is the contamination tell
metadata: 
  node_type: memory
  type: project
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-07-28T19:46:59.594Z
---

phase0c_rollout.py silently RESUMES on an existing --out parquet ("resume: N episodes already recorded" in its log) and appends only unmatched (phrase, episode) work. Any runner that reuses an out path across rolls without deleting it first produces pooled metrics averaged over stale rows from earlier runs (discovered 2026-07-28: dev8adv repair numbers were wrong; polish rows escaped because fully-new phrase sets wrote fresh files).

**Why:** the recorded n is the tell — a designed 8-phrase × 24-layout roll must show n=192; n=432/528 means accumulation. Averages move several pp (repair-260: recorded 38.6, true 34.9).

**How to apply:** always `rm -f` the --out parquets immediately before each roll (dev8 runner + roll-only script now do this). To repair contaminated history: parquet append order preserves run boundaries — segment by row order and verify each candidate slice structurally (exactly the designed phrases × uniform per-phrase counts), or filter by the run's input phrase set when the input parquet survives. Reused (phrase, layout) episodes across runs are statistically valid (frozen executor), so resume-matched work isn't wasted — only the unsegmented average is wrong. Related: [[checkpoint-sync-liveness]] discipline of verifying counts at every landing.
