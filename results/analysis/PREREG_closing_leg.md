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

## Amendment 1 (filed 2026-07-30 19:2x UTC, before any arm B sealed rolling)
**Arm D (composed router) — analysis-only, zero additional sealed exposure.**
Motivated by the val-8 strata result (v7f fully repairs hostile input on in-vocab
tasks; neutral-negative OOV) + the rules OOV premium: route by the existing
zero-count noun gate (sealed_vocab_audit.json, computed ex-ante from the 2,000
training contexts):
  - in-vocab tasks (5): CarrotOnSponge, EggplantOnSponge, GreenCubeOnPlate,
    NutOnPlate, SmallPlateOnGreenCube -> arm B (v7f) per-task cells
  - OOV tasks (7): CarrotOnRamekin, CokeCanOnKeyboard, CokeCanOnWheel,
    EggplantOnKeyboard, NutOnWheel, OrangeJuiceOnPlate, PepsiCanOnPlate
    -> arm C (rules path) per-task cells
Arm D pooled (repair condition) = concatenation of those already-planned cells;
no new rolls, routing table frozen NOW, before arm B/C repair results are seen.
Prediction: arm D repair > max(arm B repair, arm C repair) pooled — the router
beats both of its components.

## Amendment 2 (filed 2026-07-30 ~20:00 UTC, before arm B generation or rolling)
**Arm D-deploy (output-gated cascade) — deployable variant, still zero new exposure.**
Motivating audit (val-8, ledgered): every hostile val-8 input — including all four
in-vocab-task inputs — contains zero-count nouns, so an INPUT-side census gate
cannot separate "hostile rename of familiar object" from "truly novel object";
it would misroute exactly the cases v7f repairs best. v7f's own OUTPUT re-gates
cleanly: census-clean on all in-vocab tasks at every step 20-100, retains the
OOV noun ("keyboard") when no true corpus name exists.
Frozen rule: normalized-verbatim corpus hit -> passthrough; else v7f rewrite;
re-run the same zero-count census on the OUTPUT: clean -> deploy v7f phrase
(score = arm B per-task cell); retained zero-count noun -> rules path (score =
arm C per-task cell). The routing table is computed mechanically from arm B's
generated texts at generation time (before any arm B rolling) and logged.
Predictions: (i) D-oracle >= D-deploy (gap = price of text-only gating);
(ii) D-deploy >= arm B pooled on repair.

## Amendment 3 (filed 2026-07-30 ~20:20 UTC, before arm B generation; supersedes
## Amendment 2's fallback routing)
**Arm D final form (user-simplified series pipeline).** Deploy rule:
  1. Rewrite every incoming instruction with v7f (no front gate).
  2. Census the output. Clean -> deploy v7f's phrase.
  3. Retained zero-count noun -> apply the rules rewrite to v7f's OUTPUT (not the
     raw input) and deploy that.
Rationale (ledgered audit): the rules lexicon keys on canonical object names;
hostile paraphrase hides them ("black input device"); v7f recovers canonical
names ("keyboard"), so series order v7f->rules is the only order in which the
lexicon is reliably applicable. Rules-v3 keeps keyboard/wheel verbatim and only
renames analog-having objects (ramekin->bowl, juice->box) — renames v7f already
emits itself — so step 3 is expected to add only styling on keyboard-class cases.
Sealed composition (zero new exposure): clean-output tasks -> arm B cells;
retained-OOV tasks -> arm B cells IF rules(v7f_text) differs only cosmetically
(case/verb/color-adjective), logged per task at arm B gen time; any MATERIAL
fallback rewrite is flagged and that cell reported as approximated.
Dev support: series_fallback_probe.jsonl (val-8 keyboard, v7f-raw vs rules-styled
text, 24x2 each) measures the styling delta directly.
Prediction: arm D >= arm B pooled on repair; polish safety holds at the frozen
arm-B selection window (v7f polish 40.4-41.7 >= passthrough 40.6 at steps 20-40).

## Amendment 4 (filed 2026-07-30 ~21:00 UTC, before any arm A/B repair cells observed)
**Rewriter-seat comparison + census-of-record.**
1. Arm D (Amendment 3 pipeline) is composable under either rewriter seat:
   seat A = v7a-120 (arm A per-task cells), seat B = v7f-best (arm B cells).
   Both routings are produced by the same frozen rule (output census -> KEEP or
   rules-styled fallback). Seat comparison = secondary endpoint.
2. Census-of-record for all arm D routing = the Bridge-side noun counts used in
   sealed_vocab_audit.json coverage (17,297 SFT + 235k bridge annotations), NOT
   the 2,000 RL-parent census used in exploratory tables (ledgered as
   provisional). The census-of-record vocabulary is a superset, so recomputation
   can only move tasks from FALLBACK to KEEP. bridge_census.json + both seats'
   routing tables will be archived BEFORE any arm D cell is reported.
3. Primary seat prediction (filed now): seat A >= seat B on the pooled
   two-condition composite (polish decisively A: 45.05 vs 41.67; repair inside
   noise: 39.1 vs 40.4; in-vocab repair stratum identical 55.2).

## Amendment 5 (filed 2026-07-30 ~23:05 UTC; arm A mid-roll unobserved, all
## fallback cells long-rolled but not re-read for this filing)
**Headline arm D composition = user's branch spec (2026-07-30): rephrase with
v7a-120; if the OUTPUT retains a zero-count noun, the task scores as the
rules-with-Gemini arm instead.** Fallback cells are the EXISTING prereg arms —
polish: rules_pro_nominal_x12; repair: pairA_pro_x12 arm rules_v3_gemini_pro
(the frozen rules-v3 protocol executed by gemini-pro on nominal/ERT input
respectively). No new generation, no new sealed rolling beyond arm A.
Routing table FROZEN in results/analysis/armD_routing_seatA.json:
polish 5 KEEP / 7 FALLBACK; repair 7 KEEP / 5 FALLBACK (census-of-record;
unverifiable words conservatively treated zero-count: soda, teal).
Amendment 3's series semantics (rules applied to the rewriter's output) is
retained as the DEPLOY recommendation only (val-8 keyboard probe 14.6->29.2);
it is not sealed-scored (user directive: no fallback-text rolls).
Seat-B (v7f) routing will be frozen by the same procedure at arm B gen time.
Known gate blind spot (ledgered): census cannot detect brand-DROPPING
(v7a's repair emission for pepsi task says "blue can" — gates KEEP).

## Amendment 6 (filed 2026-07-30 ~23:50 UTC)
**v7f terminated at step 114 (user decision; GPU handed to v8).** Arm B
selection rule amended: best 2-rep adversarial val-8 point among the ARCHIVED
v7f checkpoints (20-100, stride 20). The step-100 adversarial roll was staged
before termination and still lands; steps beyond 100 were never durably saved.
Decision timing: made after adv 20-80 were known (best 40.36 @ step 40) but
BEFORE the step-100 pair is measured — the selection rule remains mechanical.
