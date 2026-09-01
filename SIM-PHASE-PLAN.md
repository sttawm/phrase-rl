# Sim/rollout phase — plan (drafted 2026-09-02 night, start fresh from here)

Training phase (run r1) is COMPLETE and fully archived: gemini best = iter 3
(val_avg 6.3449), claude best = its no-rules anchor (6.3097, redraw-verified),
qwen best = iter 2 (6.2045, +0.31 over its anchor). ALL r1 numbers are
proxy-scored — zero rollouts were run. The sim phase is therefore the first
ground-truth test, not just fine-tuning.

## Agreed design
- Train on the 8 classic SIMPLER tasks (the former val8). No task-level val.
- Per-iteration training sample: 16 phrases, stratified 8 natural / 5
  adversarial / 3 canonical, committed seed, fixed across iterations.
- Phrase-level HOLDOUT: 40 phrases (20/13/7), disjoint from the training
  sample, never shown to the distiller, measured ONCE after the final
  iteration — the paper's rollout-grounded headline.
- All draws from the relabeled val8 pool (365 nat / 82 adv / 15 canonical;
  see results/analysis/val8_kind_relabels.parquet). No generation → no
  sealed-set exposure.
- Scoring: 18 seeded episodes/phrase (layouts 0–17 × 1 rep), paired
  same-init deltas vs unrephrased bases.
- Passes: gemini (seeded from r1 pass_gemini iter-3 book) and qwen (iter-2
  book). No claude sim pass — its best book is no-rules = the baseline itself.
- Loop: sim_overrides (max_iters 3, no validation, patience off).

## Cost (agreed estimates)
First pass ≈ 23 GPU-h (~2,800 episodes incl. one-time bases + 40+40 holdout
pair); second pass reuses all base measurements ≈ +15 GPU-h. Two passes
≈ $15–28 on 4090-class render pods; ~1 day wall-clock with two pods.

## Open item from the last conversation (user, pre-sleep)
The original intent was that TRAIN-phase val8 be rollout-scored; it wasn't.
Optional retro-grounding: rollout-score the val8 REWRITES of gemini's
iterations 0–3 (4 books × ~96 rewrites... or the 40-phrase holdout subset per
book ≈ 4×6h) to ground the gemini trajectory in rollouts. Decide tomorrow
whether to fold this in before/with the sim loop.

## Launch checklist
1. Render pod(s): Vulkan stack (memories: ICD manifest must live in
   /usr/share/vulkan/icd.d; apt needs libegl1 libgles2; NVIDIA_DRIVER_CAPABILITIES
   must include graphics — else recreate pod). INT-ACT env for phase0c_rollout.
2. Wire stratified sampler (8/5/3) + 40-phrase holdout draw, committed seed,
   into the sim launch path; holdout bases scored alongside.
3. `config/rules_run.py gemini --phase sim --exec` with init_rules_from r1
   (verify sim_overrides picks up the pass_gemini best book).
4. Workers: sim jobs go through the same git job queue; the worker needs the
   rollout stack (NOT the score-server pods — those are gone).
