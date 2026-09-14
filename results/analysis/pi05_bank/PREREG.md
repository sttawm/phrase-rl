# Sealed-set contact log (pi0.5/LIBERO bank)

Sealed test = the 20 libero_90 tasks in splits.json (seed 20260901).
Standing rule: no LLM generation, no policy rollouts, no selection touches them
before FINAL_EVAL=1.

## 2026-09-02 — human rephrase elicitation (approved by user in-session)
- Rendered the t=0 agentview frame (init 0, seed 7) of each sealed task on a
  throwaway pod. Frames live in test_frames/ — a directory the trace pipeline
  never reads; no LLM has seen them.
- Built human_eval/rephrase_sheets.html (frame + canonical instruction + blank
  lines) for HUMAN annotators to write natural rephrasings for the final eval.
- No policy episodes were run on sealed tasks; no generated phrases exist for
  them. The human-written rephrases will constitute the sealed natural tier at
  final evaluation.

## 2026-09-02 — ground-truth end states added to the handout
- Extracted the final agentview frame of a HUMAN teleop demonstration (HF
  yifengzhu-hf/LIBERO-datasets, libero_90) for each sealed task -> end_frames/.
  Dataset content only; the policy never ran on sealed tasks.
- rephrase_sheets.html v2 shows start + successful end state per task.

## 2026-09-02 — demo clips + sharp end states; handout v3 (watch-then-write)
- Demo replay clips (32 sim states -> 4s MP4) and 256px end-state renders from
  the demos' final sim states, per sealed task. Human demo data only; the
  policy still has never run on sealed tasks.
- Handout v3 shows clip + done-frame and elicits phrasing WITHOUT displaying
  the canonical instruction (purer elicitation; canonicals remain available in
  splits.json / assets/manifest.json for the form process and the answer key).
- human_eval/assets/task_NN/{clip.mp4,start.png,done.png} + manifest.json for
  the external form-building process.

## 2026-09-03 — AMENDMENT: canonical-only screening of the sealed 20 (approved by user in-session)

WHY. Nearest-neighbour evidence suggests roughly half the sealed tasks may be uncompletable by the
policy under ANY phrasing. Two harms follow from not knowing: (a) human annotators spend effort
writing rephrasings for tasks where no wording can matter; (b) floor tasks INFLATE a pooled
rephrasing benefit — in the train bank, mean delta vs canonical at the floor is +4.8pp (natural),
+5.6pp (adversarial), +9.0pp (oracle) purely because canonical sits near 0 and any lucky episode is
upside. A naive pooled sealed mean would therefore show "rephrasing helps" driven by floor noise.
Text-side inspection cannot substitute: among train tasks, WITHIN-scene variance of canonical
success is 103% of total variance, and 9 of 19 multi-task scenes contain both a 0% and a >60% task.
Completability is a property of policy x scene, not of wording.

WHY THIS DOES NOT BREAK THE SEAL. The seal protects RULE CREATION. Both rulebooks
(rulebooks/in_plus_ood_v1.md, rulebooks/ood_only_v1.md) are frozen and committed BEFORE this
screening (commits 50e0b510, d1e3a935, b4569e74). Screening cannot leak backwards into them.

PROTOCOL (fixed before the run; deviations must be logged as a further amendment).
1. Arm: the CANONICAL instruction only. No rephrasings of any kind are generated, applied or rolled
   on sealed tasks in this step.
2. Screening window: init states 50-69 (n=20 per task). The final evaluation reports on the DISJOINT
   window 30-49. Screening numbers are never reused as the evaluation's canonical baseline — both
   arms at final eval are measured fresh on 30-49. (Selection on canonical LEVEL does not bias the
   within-task DELTA when both arms are re-measured; reusing screen numbers as the baseline would.)
3. Drop rule, fixed in advance: a sealed task with 0/20 canonical successes on the screening window
   is assigned to the FLOOR stratum. Floor-stratum tasks are excluded from human rephrase collection
   and from the primary analysis.
4. Reporting duties: the count and identity of floor tasks are reported; the primary analysis is
   explicitly conditional ("among held-out tasks the policy can perform at all"); the floor stratum
   is reported separately rather than silently dropped. Any unconditional number, if quoted, is
   labelled as such.
5. Budget: 20 tasks x 20 inits = 400 episodes, one pod, ~2h.

EXPECTED. 9 of 20 tasks have same-scene neighbours below 20% canonical; the STUDY_SCENE cluster
(73, 80, 83, 85, 86, 88) is the most at risk, and sealed 86/88's only neighbour (task 89) scored 0%
canonical and 0% best over 21 searched phrasings.

### 2026-09-03 — CORRECTION to the screening amendment (window 50-69 does not exist)

The amendment above specified screening on init states 50-69. LIBERO tasks expose exactly 50 init
states (indices 0-49) — verified on all 20 sealed tasks, every one reports 50. The first screening
attempt therefore failed with IndexError before rolling ANY episode; zero sealed episodes were
executed under the impossible window, so nothing is contaminated and no sealed measurement exists.

CORRECTED PROTOCOL (everything else in the amendment stands unchanged):
- Screening window: init states 0-19 (n=20 per task).
- Final evaluation reporting window: init states 30-49 (n=20 per task). DISJOINT from screening;
  20-29 is left unused as spare.
- This matches historical practice in the four-tier program (screens drawn low, reporting drawn
  high) and preserves the amendment's core requirement: screening numbers are never reused as the
  evaluation's canonical baseline, and both arms at final eval are measured fresh on 30-49.

## 2026-09-04 — val tasks screened on the same protocol (user-approved)

WHY. Partial sealed screening showed 5 of the first 9 completed tasks at 0/20 canonical and two more
at 5%, i.e. roughly half the sealed 20 may carry no dynamic range. That would leave ~6-9 usable
tasks, at which point even a 10-rewrite x 20-init design has an MDE near 5-6pp against a predicted
rulebook-vs-unconstrained effect of ~1.6pp. The 15 val tasks are the same population, untouched, and
were only being held back because human rephrasings exist for the sealed 20 alone.

PROTOCOL. Identical to the sealed screen: canonical-only, screening window inits 0-19, reporting
window 30-49, drop at 0/20, floor stratum reported separately. 15 tasks x 20 inits = 300 episodes.

STATUS OF VAL AFTER THIS. Screening does not promote val into the test set by itself. Whether
surviving val tasks JOIN the evaluation (with human rephrasings collected for them) is a separate
decision to be recorded here before any rephrasing of val tasks is generated or rolled.

## 2026-09-04 — val n=20 re-screen COMPLETE (salvaged from pod; supersedes "unknown" in PI05-SCREEN-FINDINGS.md)

The 3-shard direct run on ps1 did NOT die. All 15 val tasks completed at n=20
(300 episodes). Raw episodes committed to results/analysis/pi05_bank/screens/
(val_screen_shard{0,1,2}.jsonl, plus sealed_screen.jsonl for the sealed 20) so the
data no longer lives only on a pod. PI05-SCREEN-FINDINGS.md section 2 recorded the
state as unknown/possibly-OOM'd; that was correct from the machine that wrote it
(no pod SSH) but is superseded here.

VAL n=20 result: 47=100%, 67=100%, 20=80%, 0=5%, 13=5%, and TEN tasks at 0/20
(3, 4, 5, 6, 27, 42, 49, 63, 75, 87). This closely reproduces the n=10 prior in
val_canonicals.parquet (20 was 90% at n=10, 80% at n=20; 0 and 13 were 10% at
n=10, 5% at n=20) — no task changed stratum.

COMBINED OOD PICTURE (sealed 20 + val 15, both screened canonical-only, inits 0-19):
- dynamic range (>=80%): 51 (sealed, 100), 47 (val, 100), 67 (val, 100),
  30 (sealed, 85), 20 (val, 80)  -> 5 tasks
- 1/20 = 5%: 7, 43, 53 (sealed), 0, 13 (val)                                -> 5 tasks
- hard zero 0/20: 25 tasks (15 sealed, 10 val)
Floor rate is 75% sealed / 67% val — statistically indistinguishable, which
supports the split having been drawn fairly even though the yield is poor.

## 2026-09-04 — FINAL_EVAL design, fixed before any sealed generation runs

All three rulebooks are frozen and committed (in_only_v1 a3731fbb, ood_only_v1
cc0fd77e/6c783a15, in_plus_ood_v2 = rules_bank_v1). Screens are complete for the
sealed 20, val 15, and (running) the in-finetune reserve. Nothing below was chosen
after seeing any rephrasing measurement, because none exists yet.

WHY THE COMPOSITION CHANGES. The sealed-20 alone yields 2 tasks with dynamic range
(75% of it is 0/20 floor). All 90 libero_90 tasks are already allocated and every
train task carries generated phrases, so NO additional held-out OOD task exists.
Held-out range can therefore only come from the untouched in-finetune reserve.

EVALUATION SET (final; each task's stratum fixed by its canonical screen at n=20,
inits 0-19):
  A. IN-FINETUNE, canonical >=90%      -- up to 12 from the 29-task reserve
  B. OOD with range (>=80%)            -- 51, 30 (sealed); 47, 67, 20 (val)
  C. OOD marginal (1/20 = 5%)          -- 7, 43, 53 (sealed); 0, 13 (val)
  D. OOD hard floor (0/20)             -- 5 tasks, seeded random sample, kept
     deliberately as a negative control: train evidence says ~1 in 5 floor tasks
     is ever moved by any phrasing, so this stratum tests whether rules rescue
     the unrescuable. It is reported separately and never pooled into a headline.
Val tasks join the evaluation. Justification: with no loop there is no selection
for val to protect, both sets were screened under one protocol, and their floor
rates (75% sealed / 67% val) are statistically indistinguishable. Sealed and val
origin is recorded per task so any analysis can split on it.

MEASUREMENT
- Reporting window: inits 30-49 (n=20). DISJOINT from the screening window 0-19.
  Screen numbers are never used as a baseline; the canonical is re-measured fresh.
- Grid: {claude-opus-5, gemini, qwen} x {adversarial, natural, original}
  x {no rules, in_only_v1, ood_only_v1, in_plus_ood_v2}, plus the un-rephrased
  base of every tier as the baseline row.
- Bases per task: 4 natural + 3 adversarial + 1 original on strata A and B;
  4 total on strata C and D (they carry little information per episode).
- Identical rewrite strings are rolled ONCE and shared across arms (key is
  (task, phrase, init)); dedup is recorded so arm composition stays auditable.

GENERATION (recipe per PHRASE-GENERATION-RECIPE.md, unchanged from the pi0 work)
- Naturals: generate.md NATURAL block, IMAGE attached, three authors balanced per
  task (gemini-pro-latest / claude-sonnet-5 / Qwen3.5-9B), temp 1.0,
  anti-repetition carried ACROSS authors, then the SEMANTIC judge
  (judge_naturals_v2.py) -- never a lexical/word-overlap gate.
- Adversarials: ERT_PROMPT (build_sealed_assets.py), gemini-3.5-flash, temp 0.8,
  IMAGE attached, CoVer release few-shots. Distinct generator from the training
  attacks (which came from generate.md text-only), preserving held-out discipline.
- Traces: per (task, phrase), image-conditioned, generated FROM THE BASE PHRASE,
  canonical-leak gate enforced (gen_pi05_traces.py).
- Every kept line is preflight-printed and committed before rollouts begin.

PRIMARY ANALYSES (fixed now)
1. Per-stratum mean success by grid cell (the A31-style table).
2. Paired rulebook-vs-no-rules within (task, base, applier).
3. Catastrophic-collapse rate: fraction of rewrites >=40pp below their own
   canonical, by stratum -- the metric the rulebooks' own evidence predicts.
A pooled mean across strata is NOT a headline: strata C and D cannot move, and
including them would dilute any effect toward zero by construction.

## 2026-09-14 — AMENDMENT: rules distillation from single-edit pairs, sealed set v2 (approved by user in-session)

SUPERSEDES the 2026-09-04 22-task FINAL_EVAL set for this experiment. Written before any
test phrase, trace, rulebook or rollout for it exists.

TASK DRAW (scripts/draw_libero_sealed_v2.py -> sealed_v2_draw.json; seed 20260914)
- Two pools: in-finetune = the 40 finetune tasks (spatial/object/goal/10) minus libero_10/5
  (855 probe episodes exist on it; splits.json reserve_exclusions); out-of-finetune = all 90
  libero_90 tasks. One numpy default_rng(seed).permutation per pool, recorded in full.
- Walk each permutation in order and accept 10, skipping only FLOOR tasks: 0 successes of 20
  with the canonical string on inits 0-19, seed 7, standard harness. Floor status comes from
  canonical episodes already measured under that exact protocol (sealed-20 screen, the n=50
  rounds restricted to inits 0-19, the four-tier boards) or from sealed_v2_screen.jsonl
  (320 fresh canonical episodes on the 16 permutation-prefix tasks that had none). A task
  with no data halts the walk until screened; the order is never skipped or re-drawn.
- One post-draw skip, decided after the in-finetune walk was seen: libero_goal/7 holds 103 of
  the 426 single-edit pairs (24%, the stove ladder); it is treated like a FLOOR task and the
  replacement is the next task in the recorded order (libero_10/4). Recorded in the draw file.
- RESULT. In-finetune (10): spatial/2, /3, /6, /7; goal/0, /1, /6; object/1, /6; libero_10/4.
  Out-of-finetune (10): 90/60 (8/20), 90/70 (12/20), 90/46 (20/20), 90/68 (20/20), 90/54
  (15/20), 90/7 (1/20), 90/59 (5/20), 90/56 (18/20), 90/43 (1/20), 90/35 (2/20). Four of the
  ten out-of-finetune tasks are near-floor (<=5/20); they stay, per the rule, and every
  analysis is reported split by half and by canonical screen level.
- Disclosed, not filtered (task-level filtering only): sealed goal/7's canonical "turn on the
  stove" is verbatim libero_90/44's canonical, and sealed object/6's is verbatim
  libero_90/51's; both non-sealed tasks' pairs stay in the evidence. Sealed in-finetune
  tasks share scene and template with training tasks (object/*: one scene, one template):
  the in-finetune result is within-scene-family transfer, and the probe design of rounds
  2-8 was adaptive across all in-finetune tasks.

EVIDENCE (scripts/build_libero_distill_evidence.py -> distill_evidence_v2.{csv,md})
- Single-edit pairs only: results/analysis/pi05_bank/single_edit_pairs.parquet (every pair of
  phrases on one task differing in one contiguous token span, both sides n=50 on inits 0-49).
  Every pair on a sealed task is removed; nothing else is. Nulls stay in by design.
  Kept: 369 pairs (104 with |delta|>=18 pp and p<0.05) -> 271 phrases on 58 tasks
  (26 in-finetune, 32 libero_90), flattened to task / category / phrase / success % / n.
- The distiller prompt is prompts/distill_minimal.md verbatim (tag icra2027) with the table in
  its {{evidence_file}} slot (distill_prompt_v2_filled.md). Output may be bare numbered rules
  or RULES/RATIONALE; the apply path strips everything but the rules.

DISTILLATION
- Draw 1 first, end to end (user decision 2026-09-14: complete one apply+eval loop before
  draws 2 and 3, in case of failure or credit exhaustion). Each draw is a fresh, stateless
  Claude Fable multi-agent workflow (script committed as provenance) with no access to sealed
  phrases; draws 2 and 3 repeat it with the identical prompt and evidence.

TEST PHRASES, TRACES, APPLIERS
- 10 natural rephrasings per sealed task (200), gemini-pro-latest, image-conditioned on the
  task's first frame, NATURAL block of paper/prompts/prompt_generate_rephrases.txt only, plus
  12 human phrases from results/human_naturals/a39_human_phrases.parquet (4 per register) as
  style examples labelled as coming from a different robot. NO semantic judge; dedup and
  canonical-identity drop only. FINAL_EVAL=1; every kept line preflight-printed.
- Traces: one per (task, phrase), gemini-3.5-flash, temperature 0.4, image-conditioned, CoVer
  USER_TEMPLATE, generated from the incoming phrase only; canonical-leak gate.
- Arms (round 1): no-rephraser (the natural phrase), scaffold (paper/prompts/scaffold_rule.txt
  as the one-rule book), rulebook draw 1. Applier: gemini-pro-latest, thinking budget 16384,
  greedy once per (book, phrase), prompts/rules_loop/apply.md with trace + rules + phrase.
  Claude applier only if time allows.

ROLLOUTS AND ANALYSIS
- n=50: every unique (task, phrase) over inits 0-49, 1 rep, seed 7; identical strings roll once
  and are attributed to arms at analysis time (dedup ledger). Git job queue run p_v2, legs
  v2r1*, scripts/stage_libero_v2_jobs.py -> rules_loop_worker.sh -> rules_loop_jobs.py
  libero_bank_eval; per-episode records kept.
- Paired-by-base sign-flip permutation of each rulebook arm vs the no-rephraser arm and vs
  the scaffold arm, pooled AND split in-finetune vs out-of-finetune (also by canonical screen
  level); draws averaged before pairing once draws 2-3 exist. No pooled headline across
  halves.
