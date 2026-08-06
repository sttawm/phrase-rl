# Reward-exam reconciliation: the 92% vs 68% gripper cells (2026-08-06 audit)

Multi-agent forensic audit (16 agents, 12/12 adversarial verifications confirmed).
Run: wf_2adedff6-ae7. All numbers independently re-derived from raw artifacts.

## Provenance correction

The "~92% coarse sign-accuracy cell" cited in `aggregate_club_ranking.py` does
NOT come from `chunk_reward_bakeoff.py`. It is `fc_grid.json
acc_grip[F=4][C=20] = 92.35%` (`scripts/make_fc_grid.py`, B=400), computed on
the **68 native gate-zero pairs** (`gate_zero_pairs.json`: n>=200
rollouts/phrase, |delta|>=6pp, z>1.96; realized gaps median 27.1pp, 87% >=15pp)
over the 4 native exam tasks (spoon 28, eggplant 18, stack 14, carrot_plate 8).
Exact recompute: 92.3529% (diff 0.000000). Cell heterogeneity: spoon pairs run
at effective C=10, eggplant at effective F=1; spoon+carrot score 100%, stack
92.9%, eggplant 76.7%.

The "~68%" is the fine-exam far_15+ grip cell (1470 pairs, 77% OOV-task pairs).

## Decomposition of the 23.4pp gap (92.2 vs 68.8, both reproduced)

- **~10pp: task composition.** One task, `widowx_coke_can_on_plate_clean`
  (384/1470 pairs), scores **34.5% — below chance** — and carries 53.5% of
  grip's total error mass. Restricting far_15+ to the 92%-cell's 4 native
  tasks: 68.8 -> 78.6. Dropping coke_plate alone: -> 80.9.
- **~10pp: pair difficulty + label certainty.** Gap/conf-matching the native
  slice to the gate-zero statistics (gap>=25, conf>=0.99): 78.6 -> 88.2.
  Fine-exam labels use gt_n=36 (vs n>=200), mechanically deflating ~2pp.
- **~4-7pp residual:** phrase style / context-bank differences.
- **~0pp:** budget (C16 vs C20), estimator aggregation, metric definition.

## The coke_plate mechanism (new finding)

On coke_can_on_plate the raw grip signal is **inverted** (Spearman vs ground
truth -0.34): grip over-ranks attribute-laden verbose phrasings
(corr(n_words, grip_rank) = +0.69) while true success correlates negatively
with length (-0.28). Wrong votes ride tiny raw margins (median |diff| 0.0049
vs 0.0136 on correct pairs), amplified by the rank transform. 9/10 sampled
majority-wrong pairs are attribute-heavy-over-plain flips.

Mirror image: **ens100 is anti-correlated on spoon (far 41%) and stack_cube
(47%)** — both native training-family tasks. Every design has task-level blind
spots; grip's and the ensemble's are complementary.

## Design-choice verdict

- **Search** (native tasks, F=4 C=8-10): grip was the measured-BEST design
  (native far F4_C10: grip 74.0 > c4b 71.2 > z50g50 64.6 > z75g25 58.2 >
  ens100 54.0). Not suboptimal.
- **RL** (C=5 F=4): effectively tied — grip best on the native-proxy slice,
  worst on the coke-dominated pooled exam, within ~1pp of best under
  equal-task weighting. Best all-rounder: c4b (0.25 ens + 0.75 grip) by
  0.3-0.7pp task-macro — inside bootstrap noise.
- **v11 consideration:** the coke_plate-style inversion (verbosity-rewarding)
  is grip's one catastrophic mode; a c4b-style blend or a per-task sanity
  probe would hedge it.

## Paper guidance

Cite the 92% only with its true provenance (native gate-zero pairs, median
gap 27pp, C=20). Do not compare accuracies across fine-exam buckets (bucket
composition is task-confounded; every design shows non-monotone gap curves for
this reason). Valid within-bucket claims: budget trends (episodes > frames;
accuracy rises with C) and per-task design comparisons.
