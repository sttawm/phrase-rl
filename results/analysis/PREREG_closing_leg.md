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

## Amendment 7 (filed 2026-07-31 ~00:30 UTC, BEFORE rules-v4 exists — derivation
## workflow running, no rules text seen, no generation, no rolling)
**Arm E: rules-v4 (user-directed, 2026-07-31).** A new rules prompt derived by
isolated agents from ONLY: b4_phrasing_rules_v3.md + the training-data broad
search (search_boards.jsonl, transform_probe.jsonl, freeform_insights.jsonl,
club_phrase_ranking.parquet). No sealed data enters the derivation. Same
executor-side protocol as rows 12/17: gemini-pro-latest, temp 0.2, thinking
16384, sealed traces; TWO conditions — nominal input (vs rules_pro_nominal_x12
34.8) and ERT input (vs pairA_pro rules_v3_gemini_pro 31.6); 12 tasks x 24
layouts x 12 reps each. This is user-directed additional sealed exposure,
amending the final-exposure clause; phrases preflight-inspected before rolling.
Predictions (filed now): (1) v4-ERT >= v3-ERT 31.6; (2) v4-nominal >= v3-nominal
34.8; (3) gains concentrated on OOV tasks (added training-data coverage + the
strengthened in-corpus-noun rule). Rolls run on pod8 after the Tier-A leg.

## Amendment 8 (filed 2026-07-31 ~01:20 UTC)
1. **v7f step-0-polish and step-100 val-8 rolls DROPPED** (user: v7f no longer of
   interest; pod7 repurposed). Consequence: arm B selection FINALIZES on the
   measured set {0,20,40,60,80} -> best 2-rep adversarial = **step 40 (40.36)**.
   Context datum: the true step-0 baseline measured 39.58 — v7f's adversarial
   band is flat around its own init; v7e's 20 steps carried the earlier gain.
2. **Arm A repair leg parallelized to pod7** (identical protocol and phrase
   parquet; pod5 keeps polish). Cross-pod drift covered by the prereg anchor
   clause: pod7 rolls a 1-task originals re-anchor (cube_on_plate, 24x3) after
   the repair leg. Pod5's runner will be stopped after its polish leg to prevent
   a duplicate repair roll (watcher armed).

## Amendment 9 (filed 2026-07-30 ~23:05 UTC — verified clock)
**Tier-A sibling leg SUSPENDED by user before any arm completed.** Zero sibling
rows were produced or observed (anchors arm killed mid-roll, no results file).
Rationale (user): the rules were derived on the primary executor; rolling them
on the sibling conflates rule quality with derivation-target mismatch and is
not in line with the current goal. The leg may return later in its fair form —
rules/oracle re-derived natively on the sibling (the oracle re-derivation
trigger clause already anticipated this). Pod8 freed; it will run the arm E
(rules-v4) sealed rolls once phrases are generated and preflighted.

## Amendment 10 (filed 2026-07-31 ~14:30 UTC, before any arm F generation)
**Arm F: frozen Qwen + rules-v4, both conditions (user-directed).** The no-API
deployment test: the rules-v4 protocol executed by the FROZEN local Qwen
(same sft17k_rules_gen path as the v3-era rules_v3_gemini_trace row, only
--rules-path changed to b4_phrasing_rules_v4.md; nominal condition = same tool
fed a nominal-in-the-input-slot assets copy). Gemini traces, greedy, 12x24x12
per condition. Predictions: (1) F-ERT >= rules_v3_gemini_trace 31.48;
(2) the OOV-concentration signature replicates; (3) F <= the gemini-pro arm E
on both conditions (executor-strength ordering). Rolls: pod5 (ERT) + pod7
(nominal); pod8 retires with nothing queued.

## Amendment 11 (filed 2026-07-31 ~15:00 UTC, before arm G generation)
**Arm G: Claude Fable 5 + rules-v4, both conditions (user-directed).** Phrases
generated by the Claude agent applying b4_phrasing_rules_v4.md verbatim to the
sealed inputs + gemini traces (same lineage as the v3-era rules_v3_claude_agent
row, 30.99). Greedy single output per task, both conditions, 12x24x12 rolls
chained behind the arm F legs on pods 5/7; pods self-stop (runpodctl, foreground
per footgun memory) after verified push. Predictions: (1) G-ERT >=
rules_v3_claude_agent 30.99; (2) OOV-concentration signature replicates;
(3) G tracks arm E (gemini-pro) more closely than arm F (frozen Qwen) does —
protocol-following fidelity, not raw scale, is the binding variable.

## Amendment 12 (filed 2026-07-31 ~17:00 UTC, before arm H rolling)
**Arm H: Claude Fable BARE (no rules), adversarial condition (user-directed).**
Generated by a FRESH isolated Claude agent given only the ERT instructions +
scene one-liners — no rules text, no session context (the interactive agent has
internalized rules-v4, so bare generation was delegated to an uncontaminated
context; generation transcript retained). 12x24x12 on pod5, chained after its
arm G leg, then the pod self-stops. Predictions: (1) H < arm G ERT (the rules
add value over bare Claude); (2) H > gemini_pro_bare 27.78 (bare Claude
already writes canonical short imperatives — visible in the phrases: no
wrappers, brands kept); (3) H's misses concentrate where mined knowledge is
required (family overrides "cube" vs emitted "block", missing color anchors).

## Amendment 13 (filed 2026-07-31 ~17:30 UTC, before the train-only rules exist —
## derivation workflow running, no protocol text seen)
**Arm I: rules-vT (TRAIN-ONLY derivation), both conditions (user-directed).**
Purpose: the apples-to-apples partner for the RL arms — both see ONLY training
data. Derivation: isolated multi-agent workflow restricted to the four mining
artifacts (search_boards, transform_probe, freeform_insights,
club_phrase_ranking) + a scrubbed standalone corpus-vocabulary file; NO rules-v3,
NO val-8 evidence; a dedicated isolation auditor hunts imported knowledge.
Execution: gemini-pro-latest, temp 0.2, thinking 16k, sealed gemini traces
(provenance-matched to arms E/rows 12-17, isolating the RULES delta).
12x24x12 per condition, legs appended to the pod5/pod7 chains before self-stop.
Predictions: (1) vT <= rules-v4 on both conditions (v4 additionally holds val-8
certified knowledge); (2) THE HEADLINE TEST: vT vs v7a-120 (31.97 nominal /
30.12 ERT) — identical data exposure, mining vs RL; prediction vT >= v7a-120
on nominal (conservatism transfers through rules more cheaply than through
weights); (3) OOV-concentration signature replicates.

## Amendment 13 — OUTCOME (resolved without rolling, 2026-07-31 ~18:30 UTC)
**Arm I resolved BY IDENTITY: the train-only protocol's verdict is pass-through
on all 24 sealed inputs.** vT-nominal phrases are byte-identical to the anchors
originals (verified) -> by CRN determinism its cells ARE anchors_x12: pooled
**36.08**. vT-ERT differs from the passthrough arm only in case/final-period ->
~26.6 (exact roll available on request; not spent).
Predictions: (1) vT <= v4: CONFIRMED (36.08 < 37.18 nominal; ~26.6 < 33.30 ERT).
(2) vT >= v7a-120 on nominal: CONFIRMED BY IDENTITY (36.08 > 31.97) — mining
training data taught "do nothing," which beats RL's rewriting on good inputs.
ERT side: vT ~26.6 << v7a-120 30.12 — RL extracted repair value that mining
could not, WITH THE LEDGERED CAVEAT that exposure was not identical there: the
RL tier mix included CoVer ERT rewordings of parents; the mining program never
posed the repair problem. The clean decomposition for the writeup:
mining's edge = knowing when not to rewrite; RL's edge = repair learned from
adversarial-tier exposure; rules-v4 = both (val-8 evidence supplies repair).
No pod time spent; pod 5/7 chains unchanged (F -> G (-> H) -> self-stop).

## Amendment 14 (filed 2026-08-01 ~21:40 UTC, before generation of any cell)
**Five ladder-completion cells (user-directed): bare-nominal x3 (Qwen, Gemini-
Pro, Claude) + rules-v3-nominal x2 (Qwen, Claude).** Protocols matched to their
column families: GREEDY single phrase per task (ladder comparability), succeed-
framing bare prompt identical in intent to the bare-adv rows; v3 cells use the
frozen b4_phrasing_rules_v3.md verbatim. Claude generations via fresh isolated
agents (bare = zero rules exposure). 12x24x12 each. Predictions: (1) all three
bare-nominal cells land BELOW originals 36.08 (untutored strong models damage
good instructions — the conservatism law at model level); (2) Claude+v3-nominal
~ gemini+v3-nominal 34.78 (fidelity parity of strong executors); (3) Qwen+v3-
nominal < Qwen+v4-nominal is NOT predicted — v3's simpler protocol may hurt the
weak executor less than v4 did (open question, no direction filed).
A sixth OPTIONAL arm (human-phrasing simulation, sampled K=4 x 24 x 3, user's
prompt) awaits user scope decision; not generated yet.

## Amendment 12 — ADDENDUM (2026-08-01 ~23:30 UTC): arm H regenerated as v2
User-caught conditioning gap: v1's isolated agent received one-line scene
summaries instead of the FULL Gemini traces every other arm gets. v1's roll
was killed at ~90% WITHOUT any success statistic ever being computed or read
(no merge occurred) — discipline clean. v2: 12 independent isolated agents
(zero rules exposure, one per task — matching the per-call structure of the
API arms), each receiving the unified bare template (rules-v4 skeleton minus
rules) + the full sealed trace + ERT instruction. Notable v1->v2 diffs from
trace conditioning: "green block"->"green cube" (x2), pepsi gains "yellow";
coke_on_wheel regressed to "red can in the tire" (brand-drop + double family
violation) — bare-Claude's untutored noun choices remain the predicted misses.
