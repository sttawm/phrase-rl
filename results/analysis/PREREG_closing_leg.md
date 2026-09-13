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

## Amendment 15 (filed 2026-08-02 ~00:30 UTC)
1. SCOPE CUT (user): the three bare-nominal cells and the human-phrasing
   sampled arm are DROPPED — the originals reference is adopted as the human
   baseline for the nominal condition. The generated-but-never-rolled
   gemini_pro_bare_nominal phrases are discarded.
2. Remaining Amendment-14 cells: Qwen+v3-nominal and Claude+v3-nominal only.
3. NEW CELLS (user-directed): Qwen + rules with ACTUAL reasoning —
   enable_thinking=TRUE for the frozen 9B executing rules-v4, both conditions
   (12x24x12). Motivation: the executor-fidelity story predicts thinking helps
   the UNFAITHFUL executor where it demonstrably did nothing for the faithful
   one (gemini tournament 0.6pp spread). Predictions: (1) Qwen+v4+think beats
   no-think Qwen+v4 on both conditions (30.06 ERT / 27.81 nominal); (2) it
   remains below the strong executors; (3) if it fails to beat no-think, the
   9B's unfaithfulness is a capacity problem, not a deliberation problem.
4. PROVENANCE CAVEAT ledgered for the "so high" frozen-Qwen bare-adv bar
   (30.96): that row was generated under the CoVer prompt (image + trace +
   CoVer's structured rephrase scaffold — build_sealed_arm_phrases mode
   'frozen'), NOT the minimal bare template the API bare rows used. The bare
   column is not protocol-identical across models; the Qwen bar is
   scaffold-assisted and multimodal. Chart footnote to be added.

## Amendment 16 (filed 2026-08-02 ~01:00 UTC)
**Provenance-fair re-run of frozen-Qwen bare-adversarial (user-directed).**
The existing frozen_gemini_trace row (30.96) was CoVer-prompt generated (image
+ structured rephrase scaffold) and is NOT protocol-parallel to the other bare
rows. New cell: qwen_bare_ert — unified bare template (rules-v4 skeleton minus
rules), text + full Gemini trace only, greedy, enable_thinking=False,
12x24x12. The CoVer-prompt row is retained in the appendix scoreboard under
its existing label; the ladder's Qwen bare slot switches to the fair row.
Prediction: qwen_bare_ert < 30.96 (the scaffold was load-bearing), plausibly
near gemini_pro_bare 27.78.

## Amendment 15 — protocol note (2026-08-02): Qwen think-cell termination
The 9B's greedy open-form reasoning does not terminate (verified: looping
deliberation truncated at 14k tokens with no answer). The think cells therefore
use an explicit termination contract appended to the prompt ("output exactly
one line starting with 'FINAL: '...") plus repetition_penalty=1.1, budget
10240. This adaptation applies ONLY to the two Qwen-think cells and is part of
their arm definition (a weak executor requires termination scaffolding to
reason at all — itself a fidelity datum). Parse: last FINAL: line; no marker =
loud fallback, ledgered per task.

## Amendment 17 (filed 2026-08-02, before v7a-340 generation)
**Arm: v7a step-0340 (final checkpoint), both conditions, TAGGED generation**
(user-directed) — identical protocol to arm A (gen_sealed_arm.py: tier tags as
trained, sealed gemini traces, greedy, tag-stripped outputs, preflight), 12x24x12
per condition. Rationale: 340 is maximally trained and its val-8 tagged numbers
(polish 44.3 / repair 36.7 deep) beat frozen Qwen on both conditions.
Predictions: (1) primary question is TRANSFER — arm A (step 120) inverted its
val-8 polish edge on sealed (31.97 < 36.08); no direction filed for 340 vs 120
(the val-8 late plateau was flat, seedset-caveated); (2) strata signature:
IV-concentrated like every RL arm; (3) repair within ~1.5pp of arm A repair
(30.12) — the repair curve was flat 120->340 on val-8.
Legs appended behind the existing pod queues (pod5: repair; pod8: polish),
generation on pod5 after the think-cell gen completes.

## Amendment 18 (filed 2026-08-02 ~15:40 UTC, mid-roll)
**Qwen v4+think NOMINAL cell dropped (user).** Only the ERT think cell is
rolled (39% complete at filing), reported as an exploratory novelty rather
than a paired comparison: with no nominal counterpart and no bare+think
control (Amendment-15 note), it speaks to whether deliberation moves the weak
executor's repair number at all, nothing more. Phrases for the nominal cell
remain archived unrolled.

## Amendment 19 (2026-08-03 ~23:20 UTC) — Rephrase-robustness eval (user-designed)
NEW SEALED EXPOSURE, approved by user in-session. Measures executor robustness
to NATURAL human phrasing variation (complement to the adversarial ERT axis),
plus the effect of rephrase fine-tuning on that robustness.
GENERATION: K=16 rephrases per sealed nominal; gemini-pro-latest (provenance-
matched), temperature 1.0, 2 sampled calls x 8 (+<=2 top-ups after casefold
dedupe); sealed frame ATTACHED, NO trace (trace carries canonical corpus vocab
= excluded knowledge); prompt = CoVer architecture with ALL rules-like content
removed (no simplicity/adverb/color/vocab-diversity instructions, no few-shot
examples), register target = "how real people actually talk"; no case/
punctuation normalization (formatting sensitivity counts as robustness).
Full prompt frozen in scripts/gen_rephrase_robustness.py. All 192 lines
preflight-printed and posted to the user before rolls begin (rolls gated
behind the v3nom queue => hours of veto window).
ROLL: LAYOUTS 0-11 ONLY (first 12 of 24; RECORDED here and asserted in the
merge; remaining 12 reserved for a later leg — subset-variance analysis on the
per-episode records decides if they are needed). 1 rep. 12x16x12 = 2,304
eps/arm, ~5.7h each. Executors: (1) juexzz/INTACT-pi0-finetune-rephrase-bridge
(the study's standard frozen executor) -> arm rephrase16_pi0rephrase;
(2) juexzz/INTACT-pi0-finetune-bridge (NO rephrase augmentation; first
appearance) -> arm rephrase16_pi0base. Same INT-ACT ev config, --ckpt swap
only. Overnight queue on pod5 gated on v3nom completion
(scripts/roll_rephrase_robustness.sh). Rules-method arms on this rephrase set:
deferred to a follow-up amendment once the two executor baselines land.
PREDICTIONS (registered before results): (a) pi0rephrase pooled DROPS vs its
nominal-only sealed 36.08 but stays well above its ERT passthrough 26.59 —
natural variation sits between canonical and adversarial; (b) pi0base shows a
LARGER robustness gap (rephrase-augmentation is exactly training for this);
(c) per-rephrase spread within a task exceeds the pooled gap between
executors — phrasing variance dominates model variance.

## Amendment 20 (2026-08-04 ~19:40 UTC) — Rules-v4 repair of the natural-rephrase distribution (user: "gemini + RULES as a rephrase")
Each of the 192 Amendment-19 rephrases is rewritten by gemini-pro-latest +
b4_phrasing_rules_v4.md (the deployment-best doc; v3 available as follow-up if
wanted), SAME family wrap as the sealed ladder (rules + sealed trace + incoming
instruction, text-only, greedy). NOTE: the per-task sealed trace's phrase-
mapping sections decompose the ERT wording, not the rephrase — identical
convention to the ladder's nominal legs. Roll: layouts 0-11 x 1 rep, executor
juexzz/INTACT-pi0-finetune-rephrase-bridge -> arm rephrase16_rulesv4_gemini,
gated behind the pi0base roll. PREDICTION (registered): rules recover most of
the natural-variation damage — pooled >= 32 (>=70% of the 26.30 -> 34.38 gap
closed); rationale: natural inputs are easier to canonicalize than ERT, where
the same pipeline reached 33.30 from a 26.85 floor.

## Amendment 21 (2026-08-05 ~05:40 UTC) — v10 LAUNCHED: RL on prompt B (CoVer-minus-rules, no image, single-line)
User: "ready to restart RL using the basic prompt (prompt B) for a fairer
comparison to our baselines." v9 STOPPED at step 87 (final latest archived:
v9_final_step87; verdict: frozen through 42, single in->into flip at 49).
v10: COLD start from base Qwen, --prompt-family bare (prompt text frozen in
cover_prompt.PROMPT_B_*; routed at ALL build sites: generation, update prefix,
probes, val, pod6 worker via PROMPT_FAMILY env). SCORING HALVED per the
fine-pair exam (user-cited: 73 @ C10xF4 -> 69-70 @ C5xF4 vs 65 @ F-halved):
--reward-contexts 5, F=4 kept, grip-pure kept (ensemble +1pp not worth
reopening a validated definition). Economics: 8 fresh + 8 replay, accum 1,
256-cand updates, projected ~10-11 min/step (~135 updates/day). Eval: pod6
stride 14 both conditions, tag-free prompt-B generation.
STEP-0 ANCHORS (pre-registered, = the prompt-B bare-baseline row): frozen BASE
Qwen under prompt B, greedy, both conditions, FULL sealed protocol; queued on
pod5 behind the rules-rephrase arm (scripts/{gen,roll}_v10_anchors.{py,sh}) ->
arms v10step0_polish / v10step0_repair. PREDICTION (registered): the causal
anchoring test — if v10's greedy/val curves move where v8/v9 froze, few-shot/
rules anchoring caused the freeze; if v10 also freezes, the anchor is the base
model's own corpus conservatism. Secondary: v10step0 cells land BELOW the
unified-bare cells (24.19 ERT) — prompt B strips the deployment-objective
sentence and the plain-name pointer those cells had.

## Amendment 22 (2026-08-05 ~07:10 UTC) — Prompt-B three-model arms (user, pre-goodnight)
Five new arms, all layouts/protocols as established:
(1-3) rr16_promptB_{qwen,gemini,claude}: each model rewrites the 192 natural
rephrases under PROMPT B (text+trace, no rules, single line, greedy);
layouts 0-11 x 1 rep on the standard executor -> rephrase_robustness.jsonl.
NOTE user wrote "same 11 layouts"; the established subset is layouts 0-11
(twelve) — carried forward unchanged.
(4-5) promptB_{gemini,claude}_ert: sealed ERT condition under PROMPT B, FULL
sealed protocol -> sealed_ladder_cells.jsonl. (Qwen's sealed prompt-B cells
are the v10step0 anchors, already queued — polish AND repair.)
GENERATION: Gemini via API (gemini-pro-latest, temp 0, gen date stamped);
Claude via 12 isolated per-task agents (arm-G/H protocol; stateless per-input
instruction; 204 outputs collected); Qwen locally on pod5 (greedy). All phrase
sets committed pre-roll. QUEUE: pod5 sessions rrules -> v10anchor -> pbq
(prompt-B queue); ETA chain ~Aug 7 midday if serial (rebalance if another pod
comes up). PREDICTIONS (registered): (a) rr16 prompt-B arms land near the
natural-rephrase floor (26-28) for all three models — no knowledge to apply,
so executor-fidelity spread compresses; (b) sealed prompt-B ERT for Claude/
Gemini lands BELOW their unified-bare cells (29.14/27.78) — prompt B removes
the deployment-objective sentence and the plain-name trace pointer those cells
had (same mechanism as the v10step0-below-24.19 prediction).

## Amendment 23 (2026-08-05 ~15:10 UTC) — rules-v4 x {Claude, Qwen} on the natural-rephrase set
Completes the per-model rules-value triangle (user): for each rephraser M in
{Gemini done 34.33, Claude, Qwen}: no-rephraser 26.30 vs prompt-B-bare M vs
rules-v4 M; the per-model delta (rules-v4 minus prompt-B) measures the rules
content's value. WRAP: identical to Amendment 20 / the ladder (rules doc +
trace + incoming instruction) for comparability with the landed Gemini arm —
NOTED: each delta therefore spans the two prompt families (rules wrap vs
prompt-B skeleton), same as the Gemini pair. Generation: Claude = 12 isolated
per-task agents (stateless per input, 16 rephrases each; sealed-ERT rules
cells already exist in the ladder); Qwen = local greedy on pod5. Rolls:
layouts 0-11 x 1 rep -> arms rephrase16_rulesv4_{claude,qwen}. PREDICTION:
executor-fidelity ordering holds on the natural set — Claude's rules arm >=
Gemini's 34.33 - 1; Qwen's rules arm lands BELOW its own prompt-B bare arm
(richer protocol hurts the weak executor, third replication).

## Amendment 24 (2026-08-05 ~19:50 UTC) — layouts 12-23 for the natural-rephrase executor pair
Completes Amendment 19's reserved second half: rephrase16 phrases x layouts
12-23 x 1 rep on BOTH executors -> arms rephrase16_pi0rephrase_lay12,
rephrase16_pi0base_lay12 (2,304 eps each). Enables the registered subset-
variance analysis (do 12 layouts suffice?). PREDICTION: each second-half
pooled lands within +/-2.5 of its first half (26.30 / 25.00) — layout-half
exchangeability. Scheduled on e8/e9 ahead of their Qwen legs (which wait on
phrase generation anyway) — zero added wall-clock for the fleet.

## Amendment 25 (2026-08-06 ~01:45 UTC) — layouts 12-23 for the Claude/Gemini natural-treatment arms
Completes symmetric full-layout coverage (with A24's executor baselines) for:
rr16_rulesv4_{gemini,claude}_lay12 and rr16_promptB_{gemini,claude}_lay12 —
same phrases, layouts 12-23 x 1 rep, standard executor. PREDICTION (given the
executor-dependent layout effect: pi0rephrase no-rewriter half-delta +4.3):
each treatment arm's second half lands ABOVE its first half by +2 to +6.
Assignment: e7/e8 immediately (consumer duty resumes after), e4/e5 as chained
follow-ons. Qwen-arm halves NOT included (user named Claude+Gemini) — one
word adds them.

## Amendment 25 addendum (2026-08-06 ~02:00 UTC) — Qwen-arm halves added for uniform-24 coverage
rr16_promptB_qwen_lay12 + rr16_rulesv4_qwen_lay12 (chained on pod5/e9 after
their current legs). On completion the natural ladder converts to a uniform
24-layout basis (refs A/E recomputed from the existing sealed anchors).

## Amendment 25 second addendum (2026-08-06 ~02:15 UTC) — Qwen natural arms OMITTED (user)
rr16_rulesv4_qwen (25% rolled) killed; both Qwen lay12 chains cancelled.
Exception: rr16_promptB_qwen (0-11) was ~75% done on pod5 — allowed to finish
(marginal cost ~1.5h; lands in the results table, not the ladder). e9 + pod5
-> checkpoint-eval consumers. Natural ladder drops the Qwen cluster; uniform-
24 conversion proceeds for Claude/Gemini + baselines only.

## Amendment 25 third addendum (2026-08-06 ~02:35 UTC) — Qwen natural arms REINSTATED at end-of-queue (user: "do Qwen. But only at the very end")
Chains: pod5 = promptB_qwen_lay12 after its current 0-11 leg (~04:15 start);
e8 = rulesv4_qwen 0-11 re-roll + its lay12, after e8's current Claude-half leg
(~07:45 start, lands ~20:00). Consumers resume after each chain. Ladder shows
the Qwen cluster as queued placeholders; converts as cells land.

## Amendment 26 (2026-08-06, pre-registered before sealed exposure)

**Sealed validation of the val-nominated v10 checkpoint.** Nomination rule,
fixed before any sealed roll: the checkpoint with the best val-8 adversarial
cell (= step 60, 45.31 on the dev probe). Two new sealed arms, standard
protocol (12 tasks x 24 layouts x 12 reps, n=3456 each):

- `v10step60_repair` — step-60 greedy rewrites of the sealed ERT instructions
  (prompt B, text-only). Compared against the step-0 anchor 29.63.
- `v10step60_polish` — step-60 greedy rewrites of the sealed nominals.
  Compared against the step-0 anchor 29.8.

**Predictions.** Repair: sealed gain positive but shrunk relative to the
val-8 +6.8 (winner's-curse correction); point prediction +2.5 to +5.5
(sealed repair 32-35). Polish: no change (29.8 +/- 2.5), consistent with the
flat val polish curve. Phrases preflight-printed to pod6:/workspace/step60_gen.log
before any roll; paper reports sealed numbers only, with the nomination rule
disclosed.

## Amendment 26 — CANCELLED (2026-08-07 ~03:50, before any sealed exposure completed)

User decision: both v10step60 sealed arms killed mid-roll (repair ~750/3456,
polish ~600/3456; partial rolls discarded unmerged, no sealed rows written).
Rationale: the completed 31-cell val curve reads as ambiguous between
"peak-at-60 then erosion" and "noisy band with late recovery (220 = 42.7)";
rather than validate a possibly-noise-nominated checkpoint, the run RESUMES
from step ~256 with continued stride-10 checkpoint evaluation. A future
sealed validation will re-nominate on the extended curve.

## Amendment 27 (2026-08-07 ~04:30) — v11 LAUNCH PLAN (pre-registered before generation)

v11: GRPO on NATURAL rephrases only. Design deltas vs v10:
- Prompt family "bplus" = prompt B + a data-shrunk mini-rules digest (9 rules,
  authored from rules-v4 + its derivation data; frozen in cover_prompt.MINI_RULES).
- Input tier: naturals ONLY (--source-mix 0,1,0), input-dropout 0.5 retained.
  Training naturals: trace-conditioned, knowledge-free prompt, authored by
  CLAUDE (K=3) + QWEN (K=3) per unique training instruction (1,361) —
  generator-disjoint from the frozen eval naturals (gemini-pro, image-cond).
  Training naturals are trace-conditioned (the policy's own observability);
  eval naturals image-conditioned (strictly harder). Preflight: side-by-side
  register stats (length, attribute rate) before training.
- Reward: c4b (0.25 ensemble + 0.75 grip rank), C=5, F=4.
- Cold start from base Qwen. Eval: stride-10 checkpoint cells, both new
  natural-input val condition (primary) and pol/adv secondaries; n=384 at
  every 20th step; stop rule: no new natural-val peak in 60 steps.
- Sealed exposure: NONE at launch; nomination + sealed arms by later amendment.

### A27 update (2026-08-07, pre-launch)

- Rules constant renamed cover_prompt.MINI_RULES -> QWEN_MINI_RULES (mechanical).
- Input mix REVISED by user before launch: 25% original / 50% natural / 25%
  adversarial (--source-mix 0.25,0.5,0.25), replacing naturals-only. ERT tier
  reuses the v10 hostile sources + ERT-derived traces (leakage firewall).

## Amendment 28 — rules-v3 on natural rephrases (3 appliers)

**Registered 2026-08-07, before any A28 rollout.** User-requested comparison:
does rules **v3** (derived from simulation + certification evidence, BEFORE the
Jul-30 training-corpus overlay made v4 conservative/pass-through-first) beat
rules v4 on NATURAL rephrases, where v4's benefit collapsed to +0.4..+2.8?

- **Inputs**: the frozen sealed natural set `ph_sealed_rephrase16.parquet`
  (12 sealed tasks x 16 Gemini naturals) — identical to the v4 arms already
  measured. No new sealed content is generated; only a different rules doc is
  applied to the same inputs.
- **Appliers**: Qwen3.5-9B (frozen), Gemini-pro-latest, Claude — mirroring the
  existing v4/prompt-B ladders exactly.
- **Protocol**: rr legs, 24 layouts uniform (two halves, 2,304 eps each),
  greedy, CRN seeds identical to all other rr arms.
- **Wrap**: `{rules_v3}\n\n---\n\nApply the rules above.\n\nTrace:\n{trace}\n\n
  Incoming instruction: {src}\n\nReply with ONLY the rewritten instruction.`
  — byte-identical to the v4 wrap except the rules file.

**Qwen arm preflight (passed, 2026-08-07 17:0x):** 192/192 rows, 0 empty,
0 echo-input, mean 8.4 words (v4: 8.1, max 12); **57.3% of cells differ from the
v4 output**, so the arms are genuinely distinguishable.

**Predictions (registered before rolling):** v3 is more interventionist
(rebuild-first), which helped on broken/adversarial inputs but is the behaviour
the v4 overlay walked back after 14/14 freeform hypotheses were refuted on the
training corpus. On *naturals* — which are already fluent — we predict v3
performs **at or slightly below v4** (-3 to +1pp vs v4's 29.41 for Qwen), i.e.
the pass-through default is load-bearing. A clear v3 > v4 result would
invalidate the v4 overlay's central conclusion and must be reported as such.

**Gemini arm preflight (passed, 2026-08-07):** 192/192 rows, 0 empty, 0
echo-input, mean 6.9 words (max 9); **71.4% of cells differ from the v4 Gemini
output**. Note the recovery: the first generation hung on an untimed API call at
150/192; completed rewrites were recovered verbatim from the run log and only
the missing 42 regenerated (`gen_rr16_rulesv3_gemini_resume.py`, 90s per-call
timeout). Recovered and regenerated cells are byte-identical in provenance —
same model, same wrap, temperature 0.

**Claude arm preflight (passed, 2026-08-08):** 192/192 rows across 12 tasks,
0 empty, 1 echo-input, mean 7.2 words (max 10). **82.3% of cells differ from the
Claude v4 arm** (the within-applier contrast this arm exists to make) and 93.8%
differ from the Qwen v3 arm (confirming applier identity, not the rules doc,
drives most phrasing). Generated by three isolated Claude agents over disjoint
64-item slices, same trace-conditioned wrap as the Gemini/Qwen arms; the first
attempt produced no output files and was rerun with the file-write made the
explicit deliverable.

## Amendment 29 (2026-09-03, pre-registered before any sealed generation) — formalized rules-loop books on the sealed grid; adversarial condition widened to 6 attacks/task

Context. The r1/r1_sim formalized rules loop is complete: per-applier rulebooks
distilled from REAL rollout outcomes (48-base training sample at n=6, probes at
n=12, evidence surfaces on gt_success). On holdout-143 (never sealed, never
train): claude-book x claude applier 39.2%, claude-book x gemini applier 43.1%,
qwen-book x qwen 35.0%, vs scaffolds 28-29% and unrephrased 20.5%. This
amendment re-runs the paper's sealed conditions with those books, replacing the
supervised-era v3/v4 docs.

Book selection rule (fixed BEFORE sealed exposure): for each applier, the book
with the highest holdout-143 mean among {its own loop's best_rules.md, the
claude pass best_rules.md}, using the qwen x claude-book cell now scoring; ties
break to the applier's own book. Books frozen at commit time of this amendment's
follow-up ("A29 books locked") before any sealed phrase is generated.

Arms (per condition, applier applies its selected book via the r1_sim loop
protocol: prompts/rules_loop/apply.md wrap, per-base traces, applier configs
frozen in results/rules_runs/r1_sim/config.json — gemini-pro-latest thinking
1024 temp 0.0, claude-fable-5 effort high, Qwen3.5-9B greedy pod-side):
  E-cl  claude applier x selected book
  E-gm  gemini applier x selected book
  E-qw  qwen  applier x selected book
No new scaffold arms (holdout established the scaffold floor; paper baselines
stand).

Adversarial condition — WIDENED (addresses the paper's stated limitation):
6 attacks/task = the original sealed ERT (comparability anchor) + 5 NEW
per task, generated by the SAME held-out evaluation generator
(build_sealed_assets.py ERT_PROMPT, gemini-3.5-flash, temp 0.8, scene image,
cover-release shots; val8_ert_gen.py import precedent), FINAL_EVAL=1, full
preflight print, one generation batch, no cherry-picking: first 5 clean
candidates per task kept (dedup vs original ERT; reject empty/multi-line).
Episode design: 24 layouts x 2 reps per phrase -> per-phrase n=48, per-arm
n=3,456 (same budget as the paper's arm). Passthrough (no-rephrase) leg for the
60 new attacks at the same 24x2 (n=2,880) so every book arm has a paired
same-phrase baseline; the original 12 keep their existing n=288 passthrough.
Traces for new attacks: CoVer template on (scene image, attack), same as
sealed_assets_gemini.parquet.

Natural condition — set UNCHANGED (the existing 12x16 ph_sealed_rephrase16),
protocol unchanged (24 layouts x 1 rep via the two-leg first/second halves,
n=4,608/arm). One fix, as flagged in the paper's limitations: MATCHED traces —
CoVer traces generated on (scene image, natural phrase) for all 192 phrases,
FINAL_EVAL=1, replacing the ERT-derived traces the old arms were conditioned
on. Applies use those matched traces.

Predictions (filed before generation):
 P1 Pooled adversarial (6-attack mean): E-cl and E-gm both beat the paper's
    best sealed adversarial repair (34.3, rules_v4_claude_ert) on the original-
    12 subset; on the widened 72-attack set both land 30-40%.
 P2 E-gm >= E-cl on the widened adversarial set (holdout transfer pattern).
 P3 Natural condition: at least one E arm beats 36.9 (rr16_rulesv3_gemini,
    the paper's best natural); matched traces do not hurt naturals.
 P4 The 5 new attacks/task are not systematically easier than the originals:
    passthrough on new attacks within +/-6pp of the original-12 passthrough
    (26.6).
 P5 Per-applier ranking on sealed matches holdout-143 ranking (gm >= cl > qw).

Exposure ledger: this amendment authorizes (a) generation of 60 new ERT
attacks + their traces + 192 matched natural traces, (b) applies of the three
selected books to sealed phrases, (c) rollout legs listed above. Nothing else
touches the sealed grid. Estimated episodes: 3 x 3,456 (adv books) + 2,880
(new-attack passthrough) + 3 x 4,608 (natural books) = 27,072.

## Amendment 30 (2026-09-03) — no-rules (scaffold) controls on the FULL widened adversarial set

A29 declined new scaffold arms, leaving the widened 72-attack adversarial
condition without a same-protocol no-rules control (the paper's prompt-B arms
cover only the original 12 attacks, under a different no-rules prompt and no
per-attack traces). User directive 2026-09-03: run them. Three arms, one per
applier, applying the loop scaffold rule ("Rewrite the instruction as a short,
plain imperative that keeps the same objects and goal.") via the same apply
protocol as A29 (apply.md wrap, per-attack traces, frozen applier configs), on
all 72 attacks, 24 layouts x 2 reps (n=3,456/arm). No new phrase generation --
the attacks and traces are the A29 assets. Prediction P6: each applier's A29
book arm beats its scaffold arm on the full 72-attack set (paired), but by
LESS than the holdout gap (book specialization to loop vocabulary).

## Amendment 31 (2026-09-03) — v2 one-shot ultracode distillations, three evidence diets, sealed 3x3

Three books, each distilled ONE-SHOT by a Fable multi-agent panel (3 lens-
distillers -> adversarial critic -> synthesizer per arm), no iterative loop:
  T  train-only: gripper-proxy evidence (4,544 rows, 220 tasks; val8/sim and
     sealed tasks EXCLUDED from context, gt columns stripped) + corpus
     statistics + mining artifacts. Critic verifies zero sim/val8/sealed leakage.
  S  sim-only: the 516 rollout-scored bank rows (8 sim tasks) ONLY; no corpus
     artifacts, no proxy rows; black-box-VLA framing. Critic attacks overfit.
  B  both: full 5,236-row bank + corpus artifacts; synthesis of both registers.
Applier for ALL sealed arms: gemini-pro-latest (thinking 1024, temp 0.0), the
strongest measured executor -- one applier so the 3 bars per condition compare
books, not appliers. Apply protocol = loop apply.md + A29 traces (matched
naturals; per-attack; nominal traces from sealed_assets for originals).
Sealed exposure: 3 books x 3 conditions; layouts 0-11 x 1 rep for adversarial
(72 phrases) and natural (192); original = 12 nominals x 24 layouts x 2.
Episodes: 3 x (864 + 2,304 + 576) = 11,232.
Predictions: P7 B >= T and B >= S on every condition pooled. P8 T > S on
sealed OOV strata (corpus knowledge travels; sim specifics do not). P9 S > T
on the original-12 attacks (sim data saw that register's difficulty directly).

### A31 correction (user, 2026-09-03 11:00): all THREE appliers (claude fable
effort high / gemini thinking-1024 / qwen greedy pod-side) apply each book ->
3 books x 3 appliers x 3 conditions on sealed; ALL 24 layouts (adversarial
72x24x1 n=1,728; natural 192x24x1 n=4,608; original 12x24x2 n=576 per arm;
~62k episodes total). Queue order: adversarial, original, then natural.
Arm-T note: the contrast-pairs file emptied under the val8 leak scrub (all
139 pairs reference val8-family tasks) and is dropped from arm T's inputs.

### A30/A31 addendum (2026-09-03): scaffold NATURAL controls, same protocol.
The v1 promptB natural cells are not protocol-matched (different no-rules
prompt, ERT-derived traces). Re-run: loop scaffold rule, matched traces, all
three appliers, the same 192 naturals, 24 layouts x 1 (n=4,608/arm). Queued
after the A31 book arms (leg prefix c32*). These replace the v1 numbers in the
scaffold|Natural row.

## Amendment 33 (2026-09-04) — sealed NATURAL set regenerated: bank prompt, three authors

Audit finding (2026-09-04): the sealed K=16 naturals (A19) occupy one narrow
register -- 18 distinct sentence-initial words over 192 phrases, length 6-12
(mean 7.8), 7-16 template shapes per 16, 2% politeness, 5% questions, zero
conditional/constraint clauses. The bank naturals, generated from a DIFFERENT
prompt, span 137 distinct first words and 3-20 words. The A19 prompt asks for
natural variation but names no variation axes and carries no anti-repetition
memory across its two calls; the bank prompt (prompts/rules_loop/generate.md)
names axes ("vary the verb, the article, the word order and the level of
detail ... not N versions of one sentence") and passes back everything already
written for the task. The narrow set compresses the natural condition (all arms
within ~4pp) and is a stimulus-side artifact, not a policy property.

REGENERATION. New sealed natural set, K=16 per task (192 phrases), replacing
nothing -- the A19 set is retained and both remain reportable.
 - Prompt: the bank's generate.md NATURAL block verbatim (text-only, NO image,
   matching the bank's channel), plus its anti-repetition block carrying every
   phrase already written for that task ACROSS authors.
 - Three authors, balanced per task (6/5/5): gemini-pro-latest (matches the
   bank generator), claude-sonnet-5 (NOT fable -- fable is reserved for the
   claude applier per user directive 2026-09-03), Qwen3.5-9B pod-side. Author
   recorded per phrase, enabling a generator x applier self-match analysis.
 - Temperature 1.0 for all three, one call per author per task.
 - Meaning-preservation gate (mechanical): a candidate is rejected if it drops
   the task's head object noun or its goal noun (stem match against the
   canonical), if it is empty/multi-line, or if it duplicates (casefold) an
   existing phrase for that task. Rejections logged with reasons; preflight
   print of every kept line (prereg requirement).
 - Diversity acceptance check BEFORE any rollout: the new set must beat the A19
   set on distinct sentence-initial words and length range; reported alongside
   the bank's numbers.
Rollout scope is NOT authorized by this amendment -- generation and the
diversity report only. Arms will be pre-registered separately once the set is
inspected.

## Amendment 34 (2026-09-04) — sealed NATURAL condition re-evaluated on the A33 image-conditioned set

The A33 audit established that the A19 natural set occupies one narrow register
(18 distinct sentence-initial words, 5.1 templates per 16, 7% politeness) and
that its compressed spread across methods is a stimulus artifact. The A33
image-conditioned set (bank prompt, scene image attached, three authors --
Gemini Pro / Claude Sonnet / Qwen3.5-9B, no meaning filter) reaches 12.6
templates per 16, lengths 5-16, 15% politeness, and carries object descriptions
and spatial references the A19 set lacks. User selection 2026-09-04: the natural
condition is re-evaluated on the image-conditioned set; the text-only variant is
retained as an artifact but not rolled.

ARMS (natural condition only; adversarial and original columns stand):
  3 books (train-only / rollout-only / combined) x 3 appliers
  + 3 no-rules scaffold controls = 12 arms.
Appliers unchanged and as used throughout: claude-fable-5 effort high,
gemini-pro-latest thinking 1024 temp 0, Qwen3.5-9B greedy pod-side; apply.md
wrap over matched per-phrase traces (generated for the new set before applies).
Protocol unchanged: 192 phrases x 24 layouts x 1 rep = 4,608 episodes/arm,
55,296 total. Leg prefix d34*.
The A19-based natural cells are retained and reportable; both sets appear in the
record, and the comparison of the two IS a result (does a wider natural register
change the ranking, or only the level?).
Prediction P10: the level drops for every arm (a wider register is harder) and
the spread between arms widens; the rollout-only book keeps its lead.

## Amendment 35 (2026-09-04) — Gemini applier thinking-budget probe (1024 vs 16384)

Every A29/A31/A34 apply runs gemini-pro-latest at thinking_budget=1024 (raised
from 128 on 2026-09-01). The paper-era sealed arms used 16384. If gemini's
applier performance is thinking-limited rather than book-limited, the whole
applier ranking is confounded by a knob, not a capability.

PROBE (queued after A34, lower priority; leg prefix e35*): the SAME book gemini
already applied -- the rollout-only book, its strongest A31 arm -- re-applied by
gemini-pro-latest at thinking_budget=16384, everything else identical (temp 0,
apply.md wrap, same traces, same phrases), on all three conditions:
  adversarial 72 attacks x 24 layouts x 2 (n=3,456)
  natural     the A34 image-conditioned set, 192 x 24 x 1 (n=4,608)
  original    12 nominals x 24 x 2 (n=576)
A34 keeps 1024 so its cells stay comparable with the A31 grid; this probe is
reported separately as a knob check, never merged into the main grid.
Prediction P11: no condition moves by more than 2pp -- the applier ranking
reflects rule-following fidelity, not thinking budget. A larger move would mean
the A31 gemini cells understate what that applier can do, and the grid needs a
budget-matched re-run before any applier claim is made in the paper.

### A34 addendum (2026-09-04): claude applier switched to claude-opus-5
Fable credits are insufficient for the A34 applies (768 calls), so the claude
applier for the natural re-evaluation is claude-opus-5 at effort high; every
other applier setting is unchanged. CONSEQUENCE, stated so it is never silently
compared: the A34 natural claude cells use a DIFFERENT applier than the A31
adversarial and original claude cells (claude-fable-5). The claude row of the
final grid is therefore not internally comparable across conditions until the
adversarial and original claude arms are re-run under opus-5 (4 books x 3,456 +
576 episodes ~= 16k), which is not authorized here. Any applier claim in the
paper must either restrict itself to gemini and qwen (whose appliers are
constant across all conditions) or carry this caveat explicitly.

### A35 addendum (2026-09-04): gemini applier moves to thinking_budget=16384 GOING FORWARD
User directive: every gemini apply from here uses the full 16384 budget (the
paper-era setting), not 1024. Applies already run at 1024 -- the whole A31 grid
and A34's gemini column -- keep their measured values and are labelled as such;
re-evaluating them at 16384 is deferred, not abandoned. A35's probe is therefore
reframed: it no longer chooses the forward setting (that is decided) but still
answers whether the 1024-era cells understate gemini, i.e. whether the existing
grid needs a budget-matched re-run before any applier claim is published.

## Amendment 36 (2026-09-04) — replicate books evaluated on all three sealed conditions

Rulebook distillation is stochastic; A31/A34 report ONE draw per evidence diet,
so a diet difference cannot be separated from a lucky draw. Two further
independent draws per diet were distilled by an Opus panel (lens order permuted,
explicit independence instruction, same evidence files) giving n=3 per diet.

ARMS: 6 replicate books (train_only/sim_only/both x r2,r3) x 3 conditions,
applied by gemini-pro-latest at thinking_budget=16384 (the new standing setting)
and temperature 0. ONE applier by design: the question is between-DRAW variance,
and holding the applier fixed isolates it; applier x draw interaction is a
separate question not funded here.
  adversarial  72 attacks x 24 layouts x 2 reps = 3,456/book
  natural      the A34 judged set x 24 x 1      = ~4,464/book
  original     12 nominals x 24 x 2             =   576/book
Total ~51,000 episodes. Leg prefix f36*.

CAVEAT that must travel with these numbers: the r1 books in the A31 grid were
applied at thinking_budget=1024, these replicates at 16384. Comparing a
replicate against its own r1 draw therefore confounds draw with budget. The
within-set comparison that IS clean is replicate-vs-replicate (r2 vs r3, same
budget), which is what the variance estimate uses; A35 separately measures the
budget effect so the r1 cells can be placed on the same scale later.
Prediction P12: between-draw spread within a diet is smaller than the
train_only-vs-sim_only gap measured in A31 (i.e. the diet effect survives).

### A36 revision (2026-09-04, user): all three appliers, r2 before r3
Scope widened from gemini-only to the full grid, sequenced by replicate so the
first complete replicate lands early:
  PHASE 1 (r2): 3 diets x 3 appliers x 3 conditions = 9 arms x 8,496 = 76,464 eps
  PHASE 2 (r3): same again = 76,464 eps
Appliers as standing: claude-opus-5 effort high, gemini-pro-latest @16384 temp 0,
Qwen3.5-9B greedy pod-side. Leg prefixes f36* (r2) and g36* (r3) so phase 2 can
be queued or dropped independently after phase 1 is read.
With n=3 draws per (diet, applier, condition) the published table can carry
between-draw error bars on EVERY cell rather than only the gemini row.

### Amendment A37 (2026-09-05): CoVer head-to-head on the sealed 12
Filed BEFORE any generation. External comparison against CoVer (arXiv:2602.12281,
released code + verifier weights at github.com/cover-vla/cover-vla).

SEALED CONTACT AUTHORIZED BY THIS AMENDMENT:
- Rephrase GENERATION touching sealed bases: CoVer's released rephrase pipeline
  (their verbatim prompt scaffold, whose trace slot our cover_prompt.py port
  preserves), 8 rephrases per base, applied to the 72 sealed adversarial bases,
  the 186 A34 judged naturals, and the 12 canonicals. FINAL_EVAL=1 and a
  preflight print of every base before generation.
- TRACES: the existing registered per-base traces (trace registry incl. the
  +264 sealed rows) are REUSED verbatim. No new trace generation. Every arm in
  the comparison (passthrough / rulebook / CoVer / combined-if-run) conditions
  on identical trace rows.
- ROLLOUTS on sealed 12: CoVer arms at the matched grids (adversarial 24x2
  committed now; natural 24x1 and original 24x2 as separate later gates), plus
  a verifier-off passthrough re-measurement (2 attacks/task x 24) inside their
  harness as a cross-check against our phase0c numbers.

EXPLICITLY NOT TOUCHED: the reserved CoVer OOD trio (redbull_on_plate,
zucchini_on_towel, tennis_in_basket) stays sealed; harness validation against
their published numbers uses their 4 in-distribution tasks only (val8 natives,
train-side, not sealed).

PROTOCOL FIXED IN ADVANCE: checkpoint = INTACT-pi0-finetune-rephrase-bridge
ONLY (their Table-3 "pi0 w/ Inst. Aug." row is the printed anchor, pending
stage-0 verification of their exact checkpoint id). Episode ids 0-23, same
bases, same traces. CoVer's internal stochasticity (8 rephrases, 5 action
samples/step, verifier ensemble) is NOT noise-pinned -- it is the method under
measurement; reps average it. Anchors precede heavy compute: (A) their 4 ID
tasks, verifier off ~= 44.0 then on ~= 65.5 (+-4-5pp; their trial counts are
unpublished); (B) our sealed 12 verifier-off vs our passthrough within CRN
noise. Either anchor failing STOPS the program before the heavy arm.

Predictions: P13 CoVer-heavy beats passthrough on sealed adversarial. P14 (open
question, no prediction): CoVer vs the rollout-only rulebook at ~40x test-time
compute difference. Combined arm is gated on the adversarial read.

#### A37 deviation note (2026-09-08, user decision): Gemini substituted for GPT-4o
CoVer's released rephrase generator calls GPT-4o; no OpenAI credential exists in
this project. Rephrase generation for the sealed bases will run CoVer's pipeline
(their generator script, batching, dedup, boot-time first-frame conditioning)
with the VLM swapped to Gemini via the OpenAI-compatible endpoint -- a one-line
model substitution, everything else theirs. Anchor A is unaffected (it consumes
their SHIPPED GPT-4o rephrase file for their tasks). Recorded as a method
deviation: the CoVer-on-sealed-12 arm is "CoVer with a Gemini rephraser", and
any rephrase-quality gap travels with that caveat.

#### A37 Anchor A result (2026-09-09): PASS
Their shipped test_pi.sh ran to completion on cv2 (their checkpoint
INTACT-pi0-finetune-bridge, verifier on, 8 rephrases, 100 trials/task, their
shipped GPT-4o rephrase file). Correction to the printed anchor: 65.5 is their
Table-3 pi0(rephrase)+CoVer row; the applicable row for the shipped script's
checkpoint is pi0+CoVer. Per-task, ours vs their Table-3 pi0+CoVer:
eggplant_in_basket 89.0/89, spoon_on_towel 44.0/40, block_stacking 50.0/51,
carrot_on_plate 53.0/48 (ID pooled 59.0 vs 57.0, +2.0pp); OOD (reference only,
trio remains excluded from our comparison): redbull 51.0/51, zucchini 27.0/41
(-14pp, ~2.9 sigma at n=100, the one outlier), tennis 92.0/91 (OOD pooled 56.7
vs 61.0). 6/7 tasks within +-5pp, two exact -> harness reproduces their stack;
program proceeds to Anchor B. Verifier-off leg of Anchor A was not run (the
shipped script covers verifier-on only; verifier-off validation is Anchor B's
role against our own passthrough).

#### A37 Anchor B, first attempt (2026-09-09): FAIL — horizon mismatch found and fixed
Their harness on our sealed 12 (k1+k2 attacks, 24 pinned layouts, verifier off)
scored 53.3 pooled vs our passthrough reference 26.1 (+27.2pp, every cell
high). Root cause: their episode loop hardcodes a 150-step horizon and ignores
TimeLimit truncation; our benchmark's envs truncate at 60 (SIMPLER
registration). 20.0pp of episodes succeeded after step 60. Fix: pin_layouts
mode now breaks on truncation (A37-horizon patch, both pods) — the head-to-head
holds the benchmark horizon fixed at 60 for every arm.
Horizon-adjusting the recorded episodes (success only if steps<=60) leaves
33.3 vs 26.1: a residual +7.2pp serving-stack offset (their integration of the
INT-ACT BridgeSimplerAdapter + their PI0 serving vs our phase0c serving), or a
host/driver effect (our 26.1 reference came from the retired r1 fleet).
Triangulation in progress before any heavy launch: (a) Anchor B re-run under
the horizon fix (their harness, cv3); (b) the SAME 24 cells rolled by OUR
phase0c on the SAME GPU (cv2). If (b) reproduces ~26 the offset is their
serving; if (b) lands near (a) the offset is host drift and the fleet-era
reference is the outlier. No heavy compute until this is read and the pass
criterion re-evaluated.

#### A37 Anchor B triangulation result (2026-09-10): offset is their serving stack
Horizon-fixed re-run of their harness (cv3): 32.3 pooled vs our 26.1 reference
(+6.2pp, FAIL at +-4). Same-GPU control — OUR phase0c on the identical 24
cells on cv2 — 25.6 at 540/576 (final number recorded when complete): the
fleet-era reference transfers; host drift is excluded. Attribution: ~+6pp
serving-stack difference in their harness.
Serving diff identified as the leading candidate: our harness serves pi0 in
bf16 (INT-ACT pipeline defaults use_bf16=True, use_amp=True: weights cast +
autocast) — the entire published grid ran bf16 — while their harness loads
fp32 with no autocast. Per user decision, the alignment direction is: make
THEIR verifier-off reproduce OUR passthrough (our grid is frozen). A/B-1
running: their harness with policy_bf16 flag (weights->bf16 + autocast + input
cast + float-before-numpy, mirroring our policy_wrapper semantics exactly),
same 24-cell anchor. Pass -> level-based A37 design proceeds with policy_bf16
in the aligned CoVer config; residual -> next knob (their forked batched
select_action path, ensemble-temp no-op check).

#### A37 serving-offset A/B-1 (2026-09-10): bf16 EXONERATED
Their harness verifier-off with pi0 served bf16 exactly as ours (weights cast +
autocast + input cast): 33.2 pooled vs 32.3 fp32 — no movement. Precision is
not the offset. Also verified by direct inspection: CoVer's harness imports
INT-ACT's own BridgeSimplerAdapter (identical preprocess/postprocess/dataset
statistics), their lerobot fork's pi0 is INT-ACT's + a default-1.0 noise_std
argument (num_steps=10 both), checkpoint config identical, sim stack identical
(sapien 2.2.2, same editable ManiSkill2_real2sim, gymnasium 0.29.1, numpy
1.26.4). Sole remaining stack difference: torch 2.11.0+cu128 (their venv) vs
2.6.0 (ours). A/B-2 (runtime bisect) running: OUR phase0c rollout code
executed inside THEIR venv on the same 24 cells — ~32 implicates the torch/
lerobot-fork runtime; ~24 implicates their eval-loop code specifically. Heavy
arm remains gated (user condition: launch only when their verifier-off
reproduces our passthrough).

#### A37 serving-offset ROOT CAUSE (2026-09-10): success-metric mismatch
Runtime bisect: our phase0c inside their venv = 26.0 pooled (vs our 24.1-26.1
band) — torch 2.11, their lerobot fork, weights, adapter, sim stack all
exonerated. Paired action traces (both harnesses, same venv, same CRN seed,
12 episodes): actions differ only at ~1e-3 from step 0 (bf16-vs-fp32
rounding, pooled-neutral per A/B-1), but EPISODE SEMANTICS differ — their
loop breaks on FIRST success (ever-success within horizon; episodes end at
step 26/31/38/57), phase0c runs to truncation and scores FINAL-STATE success
at 60. pi0 continues acting after success and can undo it: ever-success >=
end-state, +6-7pp systematic (5/12 vs 3/12 on identical episodes). Our
published grid is end-state@60 throughout; their published numbers are
ever-success@150 (their benchmark, self-consistent; Anchor A unaffected).
Alignment per user decision (their verifier-off must reproduce our
passthrough): A37-metric patch — under pin_layouts their loop runs to the
60-step truncation and scores done-at-exit. Metric-fixed anchor re-running;
pass criterion unchanged (+-4pp pooled vs 26.1). One found-and-fixed
harness-adaptation bug also recorded: the bf16 patch had swallowed their
config.device assignment into the flag branch (fp32 path crash; no completed
result affected).

#### A37 Anchor B FINAL (2026-09-10): PASS — 25.7 vs 26.1 (-0.4pp)
Metric-fixed anchor (their harness, verifier off, our 24 k1/k2 attack cells,
24 pinned layouts, 60-step horizon, final-state success, fp32): 25.7 pooled
vs our 26.1 passthrough reference. Alignment achieved: their verifier-off IS
our passthrough within CRN noise. The aligned CoVer configuration is
therefore: their released pipeline + verifier, our benchmark contract
(sealed-12 suite, our 72 attacks, episode ids 0-23, 60-step horizon,
final-state success), fp32 serving (matching the passing anchor), CoVer
internal stochasticity unpinned (the method under measurement). Heavy
adversarial arm LAUNCHED per the standing order: waves k0-k2 on cv3 (24
layouts x 2 reps each, verifier on, 8 rephrases, 5 action samples/step);
waves k3-k5 queue onto cv2 when its pair-verification shard completes.

#### A37 heavy adversarial, PRELIMINARY (2026-09-10, waves k0-k2 = 36/72 attacks)
CoVer best-config (verifier on, 8 rephrases, 5 samples/step) under the aligned
benchmark contract: 21.8 pooled over 36 attack cells (1,728 eps) vs their own
verifier-off 25.7 and our passthrough 24.5-26.1. P13 (CoVer beats passthrough
on sealed adversarial) is at half-sample heading to FALSIFIED under
final-state@60 — the verifier costs ~4pp net. Per-task structure is coherent,
not noise: helps floor tasks (carrot_on_sponge +13.5, eggplant_on_sponge
+15.6pp vs passthrough) and hurts survivable ones (pepsi -28.1, orange_juice
-22.9pp) — the same conditional-intervention structure as our in-vocab/OOV
finding, expressed inside their method. Caveats logged in advance of the full
read: their method was designed for ever-success@150 (their published gains
replicate in OUR Anchor A at that metric); a metric-ablation arm (their loop,
break-on-done at 60) would separate horizon from metric if wanted. Waves
k3-k5 complete on cv2 shortly; final read then.

#### A37 decisions + verification (2026-09-10, user)
- Combined rules+verifier arm: NOT RUN (user decision, given the verifier-
  net-negative adversarial read).
- Natural + original CoVer conditions: COMMISSIONED. Rephrases for the 186
  A34 naturals generated under the same pipeline (preflight passed); ragged
  nat waves k1-k16 + the orig wave rolling.
- Metric: user judgment is that ever-success (their convention) is the better
  metric; our historical grid cannot be re-scored (final-state only recorded).
  Going forward the harness records BOTH metrics per episode (ever_success
  field) plus a per-episode count of verifier low-confidence swaps and the
  final selected instruction, so nat/orig read out both ways and the swap-rate
  diagnostic comes free. Directionality of the adversarial read is expected to
  hold under either metric (to be confirmed on nat/orig; an adversarial
  re-roll under dual metrics is a cheap later option).
- Prompt fidelity verified byte-level against the DEPLOYED CoVer code (not
  just the July reference): get_rephrase_batch user turn (two batch sizes)
  and the system persona are byte-identical to our port. Deviation remains
  Gemini-for-GPT-4o only.
- Rephrase-pool audit for the hurt tasks: pools are goal-preserving and
  image-grounded, but rich in exactly the lexical features our minimal-pair
  mining measures as policy-toxic (destination color words on plate tasks,
  dish-for-plate, capitalization). Mechanism hypothesis: the verifier scores
  trajectory consistency, not lexical toxicity; low-confidence swaps move
  INTO toxic rewrites on survivable tasks.

#### A37 heavy adversarial FINAL (2026-09-10): CoVer 21.6 — P13 FALSIFIED on this benchmark
All 72 attacks x 24 pinned layouts x 2 reps (3,456 episodes, verifier on, 8
rephrases, 5 samples/step, final-state@60): pooled 21.6 vs their own
verifier-off 25.7 and our passthrough 24.5-26.1 — the verifier is net
-4.1pp against its in-harness baseline. The conditional structure sharpened
at full sample: hurts survivable tasks (orange_juice -25.8, pepsi -21.7pp vs
passthrough on matched cells) and helps floor tasks (eggplant_keyboard +15.8,
carrot_sponge +12.9, eggplant_sponge +10.8). P13 (CoVer beats passthrough on
sealed adversarial) FALSIFIED under the aligned benchmark contract. P14: the
rollout rulebook (~30 pooled, ~1/40th test-time compute) beats CoVer by ~8pp.
Pending: natural + original conditions (dual-metric), the ever-success
metric ablation on anchor cells, and the swap-rate diagnostics — all rolling.

#### Metric-sensitivity slice (2026-09-11): grid deltas are METRIC-STABLE — no re-roll
1,392 dual-metric episodes (our harness): originals passthrough, naturals
sample, rollout-book adversarial rewrites. Ever-vs-final gaps: +4.7 / +4.7 /
+8.0pp — arm-independent within comparisons; task-level gaps correlate
0.63-0.89 across arms (un-solving is a task property; eggplant_on_keyboard is
the outlier at 12-33pp under every arm). Conclusion: first-success rescoring
shifts fixed-instruction arms together, preserving all grid deltas — the
published grid stands under final-state@60 with this note; no re-evaluation.
Fixed-instruction hold rates 83-85% vs CoVer-originals 74%: ~10pp excess
un-solving attributable to the verifier (measured, supports the
destabilization mechanism).

#### A37 natural condition FINAL (2026-09-11): CoVer 22.8 — net-negative in ALL THREE conditions
186 naturals x 24 pinned layouts (4,464 eps, verifier on, final-state@60):
pooled 22.8 vs our no-rephraser 26.0, scaffold ~28.8, rulebooks 27.1-30.7.
Ever-success 30.6 (+7.8 churn gap; swap rate 6.5/15 — condition-independent).
The full picture, final-state vs no-rephraser: adversarial -2.9 (21.6 vs
24.5), natural -3.2 (22.8 vs 26.0), original -9.7 (26.4 vs 36.1). CoVer's
verifier is net-negative on every condition of this benchmark; the damage
scales with how good the un-intervened instruction already was — the
conditional-intervention thesis, third confirmation.

#### A37 four-arm dual-metric comparison, adversarial full-72 — FINAL (2026-09-11)
All arms COMPLETE at 72/72 bases x 24 layouts x 1 rep, dual-metric +
first-success step, base-weighted. (Correction vs the interim read: the
rules/scaffold rolls covered 60 DISTINCT applied phrases because the applier
mapped some attacks to identical outputs; pooling now expands each applied
phrase back to its bases via the A31/A29 apply tables — 72/72 covered. The
interim 29.0/35.6 and 24.0/31.4 were phrase-weighted.) final / ever:
rules-gemini 28.1/35.5 (median first-success step 29); scaffold 24.1/32.4
(31); passthrough 24.4/31.8 (32; ever72_off k0-k5, cv4+cv2); CoVer-on
20.7/28.5 (32; the x1 read replicates the x2 heavy 21.6). VERDICT
unchanged: ever-success shifts every arm ~+7pp nearly uniformly — margins
preserved; rules beat CoVer by +7.0pp under CoVer's own metric (35.5 vs
28.5) and +7.4 under final; CoVer stays below plain passthrough under both
metrics. Speed: CoVer's median time-to-first-success equals passthrough (no
speed benefit from 40-candidate selection); rules arm fastest. P14 holds
under both metrics. Raw episodes: a37_cover_runs/ever72_{on,off}_k0-5.jsonl.

#### Amendment A38 (2026-09-11, pre-registered before generation): scaffold x Original completion
The no-rules scaffold was never measured on the Original condition (the
lone hole in the 4-arm x 3-condition grid; charts show "not run"). Plan:
apply the frozen no-rules scaffold book (results/rules_runs/v2distill/
scaffold.md, unchanged) to the 12 canonical instructions with all three
appliers (claude-fable-5 effort high / gemini-pro-latest thinking 1024 /
Qwen3.5-9B greedy pod-side), via the standard A31 apply path
(gen_a31_applies.py, tag "sc", FINAL_EVAL=1), original-condition traces,
then roll every rewrite on the full 24-layout grid x 2 reps (seed 42, CRN,
final-state@60) -- the exact protocol of the A31 orig cells. No rulebook
cell is re-rolled; no sealed phrase is regenerated (the 12 nominals are
inputs, as in every prior orig apply). Predictions logged: P15 scaffold ~=
baseline 36.1 (as in adv/nat, rephrasing without rules neither helps nor
hurts originals materially); the interesting alternative is a qwen
over-edit harm case mirroring qwen x rollout-only x orig.

#### Amendment A39 (2026-09-12, pre-registered before generation): human-naturals condition
37 survey submissions (three register prompts: adult / kid / robot) yield 392
cleaned unique (task, register, phrase) rows over all 12 sealed tasks (10
invalid entries removed: placeholder junk and destination-less incompletes,
logged in build_a39_human_naturals.py output; human typos kept VERBATIM --
the register is the treatment). Plan: (1) gemini traces per (task, phrase),
standard recipe (CoVer-template USER_TEMPLATE, gemini-3.5-flash, temp 0.4,
scene image, extract_trace), trace from the HUMAN phrase, never the
canonical; (2) applies for ALL NINE draw-books (3 diets x r1/r2/r3) plus the
no-rules scaffold, by three appliers -- claude-fable-5 at effort MAX, gemini
at thinking budget 16384 (both deliberately above the A31 settings; noted as
a protocol difference), Qwen3.5-9B greedy pod-side as before; (3) rollouts
of the applied phrases plus the raw human baseline on the 24-layout grid
(rep count to be fixed with cost sign-off before rolling). Sealed phrases
enter only as apply INPUTS, as in every prior condition.

A39 addendum (2026-09-12, before any rollout): rollouts at 24 layouts x 1
repetition (n=1 confirmed by user; the metric-stability slice justifies the
single-rep design). The roll set: unique (task, phrase) over all apply
outputs + the raw human phrases, deduplicated BEFORE rolling (CRN makes
duplicate rolls byte-identical); whether to roll all 392 human bases or a
uniform subsample is decided AFTER the appliers land, when the dedup
collapse is known. Traces: all 363 unique (task, phrase) pairs traced
(image-conditioned, gemini-3.5-flash), cached by exact (task, phrase) key.

#### A38/A40 trace-regime correction (2026-09-13)
User-directed: ALL new runs use clean traces. Found: the 12 canonical rows
in the per-base trace registry were keyed to ERT-derived trace text at A31
registration (410ebc31d) — every Original-condition apply through the
registry (A31 r1 books, A36 r2/r3, A38 scaffold first pass) saw them; the
adversarial condition is trace-from-base by construction and the A34
naturals + A39 humans used correctly-derived traces. CoVer/A37 is
unaffected: its rephrase calls generate their own inline reasoning trace
per its protocol (verbatim scaffold, image-conditioned, Gemini swap = the
recorded deviation). Fix: 12 clean trace-from-nominal rows replace the
corrupted ones (a38_clean_nominal_traces.parquet); A38 scaffold applies
redone (scc: gemini 1/12, claude 3/12 rewrites changed; unchanged rewrites
keep their rolled episodes under CRN); qwen apply requeued (a38scqwo2).
A40 (queued end-of-night): all nine books x cl/ge/qwen orig applies redone
on clean traces, rolling only changed rewrites. User recalls a
pre-migration clean-trace orig rerun; no such artifacts found in git
(searched leg prefixes + registry history) — treating A40 as the
authoritative clean-trace orig record. The first-pass A38 numbers
(scaffold 30.7 vs 36.1, p=.038) are superseded by the scc read.

#### A38 FINAL (clean traces, 2026-09-13): scaffold x Originals = harm in every cell
claude 31.8 (-4.3, p=.17) / gemini 31.1 (-5.0, p=.056) / qwen 28.5 (-7.6,
p=.063) vs anchors 36.1; exact sign-flip, n=12 bases each. P15 falsified:
the no-rules rephraser damages originals, and the damage orders by edit
aggressiveness (claude kept 2 canonicals verbatim under clean traces; gemini
normalized all 12 to sentence-case+period; qwen additionally added colors
and sibling nouns -- each edit class independently measured harmful by pair
verification). First-pass (ERT-trace) numbers superseded but concordant
(-5.4 pooled cl/ge). The rulebook orig cells (-1 to +4 vs baseline) vs the
scaffold (-4 to -8) show the books' copy-by-default rule protecting
originals from the rephrase wrap itself.

#### A39 FINAL (2026-09-13): human naturals, 30 arms, all significant gains
363 unique human phrases (37 submissions, 3 register prompts), 24 layouts x 1
rep, ~53k episodes total incl. arms. RAW HUMAN BASELINE 22.96 -- BELOW the
LLM-generated naturals baseline (26.0): real human phrasing is harder for
pi0 than synthetic naturals. Every arm improves on raw humans (paired
sign-flip, n=363; all p<=0.03, rollout books p<0.0001): rollout-only draws
+3.4 to +7.6 (r3 draws strongest: s3 ~+7 all appliers, b3 +5.7 to +7.5);
train-only +0.4 to +2.5; scaffold +2.1 to +4.2. Registers (adult 24.3 / kid
23.5 / robot 23.3 raw) lift in parallel. Cells:
results/analysis/a39_human_cells.json.

#### A40 FINAL (2026-09-13): clean-trace book x orig
r1 frontier cells replicate the published ERT-era numbers within 0.7pp
(s/b/t x claude/gemini) -- the A31 Original column stands for frontier
appliers; qwen r1 cells move (b 37.0->32.5, s 28.5->31.2, t 38.0->35.7),
qwen being trace-sensitive in both directions. Replicate draws on
originals: b2 41.4/42.2/33.2 (best orig cells measured), s2/s3 32.6-34.4
(aggressive rollout draws hurt originals), t2/t3 ~37.3 (barely edit).
Note: A40 cells mix 24x2 (fresh/b31) and 24x1 (A39-overlap) rates.
Cells: results/analysis/a40_clean_orig_cells.json.
