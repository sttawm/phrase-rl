# Pre-registration: the closing-table sealed leg (filed 2026-07-30, before any rolling)

## Arms (all sealed 12-task grid, 24 layouts, 12 reps, pooled n=3,456/arm/condition)

**A. v7a step_0120** (checkpoint of record; frozen). Both input conditions:
  - polish: nominal sealed inputs + sealed gemini traces, GEN_TAG "[input: original wording]"
  - repair: sealed ERT inputs + ERT-derived traces, GEN_TAG "[input: adversarially reworded]"
  Greedy, tagged generation via the patched gen path; emitted tag prefixes stripped;
  prompts preflight-inspected before rolling.

**B. v7f best checkpoint.** Selection rule (FROZEN NOW): the checkpoint with the best
2-rep adversarial val-8 point among steps <= 140, chosen when the curve through
step 140 is measured — before any sealed rolling of arm B. Same two conditions as A.

**C. Gated-mechanical pipeline (rules-v4-lite).** Deterministic, no LLM at deploy:
  - IF input is nominal AND every content noun has nonzero Bridge-corpus count:
    pass through byte-identical.
  - ELSE: apply the rules-v3 (gemini-pro) rewrite path.
  Gate provenance: ex-ante corpus audit (zero-count rule) + train/dev triangulation
  (213-board search, 49x16 transform grid, freeform holdout). Mixed/low-count nouns
  (e.g. nut=60): nonzero => in-vocab, no exceptions.

**Anchor cell:** each pod used rolls a small originals re-anchor (1 task x 24 x 3)
for cross-pod drift calibration; only pooled numbers are headline-grade.

## Predictions (filed before rolling)

1. Composite prediction for arm C, computed from existing sealed parquets
   (in-vocab tasks from anchors_x12, OOV tasks from rules_pro_nominal_x12):
   **36.8 pooled** (+0.7 vs originals 36.1). Test: arm C lands >= originals.
2. Arm A polish lands above v6_rl (30.1) and at/above sft_v2 (24.6); repair
   condition recovers part of the ERT gap with nonzero repair on rename-class
   OOV tasks (secondary endpoint: per-task OOV repair cells).
3. Arm B repair >= arm A repair on OOV rename tasks (the keyboard-class
   capability), per the val-8 evidence.

## Secondary endpoints
Per-task strata, especially the 7 OOV tasks under the repair condition — a pooled
tie with per-task capability structure is a pre-registered live outcome.

## Tier-A generalization leg (separate, non-augmented sibling executor)
Frozen text arms (anchors originals / rules_v3_gemini_pro / oracle_confirmed
phrases) rolled on juexzz/INTACT-pi0-finetune-bridge, 12 tasks x 24 layouts x
3 reps per arm. Questions: does the 2x2 law replicate without paraphrase
augmentation; does the rename premium grow; do transferred oracle phrases hold up
(trigger rule: if the transferred oracle underperforms its source-model level by
>8pp pooled, a reduced-protocol oracle re-derivation on the sibling is justified).

## Discipline
This is intended as the final sealed exposure for the current program. All design
choices above derive from train/dev evidence; nothing may be revised after first
roll except by documented amendment. Pre-commit: report whatever lands.
