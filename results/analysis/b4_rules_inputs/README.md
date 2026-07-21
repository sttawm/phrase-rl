# B4 rule-distillation inputs — the runner's working set

Every file here is evidence for ONE reasoning task: author an explicit,
task-agnostic rule set for phrasing instructions so the frozen π0
(INTACT rephrase-finetune) executes them with maximal success.

## Files
- `01_bridge_gt_instructions.txt` — all 17,297 unique train-split Bridge GT
  instructions (π0's modal phrase distribution). Unlabeled; structure only.
- `02_oxe_paraphrases_SAMPLE60keys.csv` — readable sample (60 full clusters) of
  the 235,764 benign paraphrases π0 saw in finetuning. Full set:
  `results/phrase_artifacts/oxe_paraphrases_bridge.parquet`.
- `03_contrast_pairs_139.json` — 139 powered same-task phrase pairs with
  measured success, n, delta, z (layout-clustered). TOP EVIDENTIARY TIER.
- `04_greedy_cells_all_arms.csv` — every greedy ×12 (task, phrase, success)
  cell across all arms (n=288 each). Second tier.
- `05_sampled_pools_per_draw.csv` — per-draw phrases with success over 24
  layouts, pool-oracle winners flagged. THIRD TIER: single-draw numbers carry
  SE ≈ 10pp and oracle flags are winner's-cursed — use draw STRUCTURE
  (what kinds of phrasings win), never raw ranks alone.
- `06_mechanism_cases.md` — the curated case families with provenance.
- `07_selftraces_val8.md` — what the deployment-time Qwen self-trace actually
  provides (scene sentence + referent mapping) for each val task.

## Rule-form constraints (structural generalization guard)
Rules must be TASK-AGNOSTIC and EXECUTABLE: functions of (input instruction,
self-trace, corpus statistics). Banned: naming any val object/task
(e.g. not "prefer cube over block" but "when multiple concrete nouns fit the
traced object, prefer the one more frequent in the training corpus for that
object class"). Every rule should cite which evidence tier motivates it.

## Evidence weighting
CI-cleared pairs (03) > n=288 greedy cells (04) > pooled draw patterns (05)
> any single oracle rank. The deployable consumer of the rules is frozen Qwen
+ self-trace (07 shows exactly what the trace gives you to work with).
