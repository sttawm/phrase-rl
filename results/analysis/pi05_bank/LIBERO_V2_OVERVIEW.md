# pi0.5 / LIBERO rules-distillation experiment (sealed set v2) — overview for the paper

Written 2026-09-15 from PREREG.md, sealed_v2_draw.json, libero_v2_round1_cells.json and the
run logs. Round 4 (draw 4) completed 2026-09-15; all four draws are in the table.

## 1. Setup

- Policy: frozen pi0.5 fine-tuned on the 40 LIBERO tasks of libero_spatial / libero_object /
  libero_goal / libero_10 ("in-finetune"). The 90 libero_90 tasks are never trained on
  ("out-of-finetune"). Success = the environment's BDDL goal predicate; an episode ends at the
  first success. Step budgets: spatial/object/goal 300, libero_90 400, libero_10 520 (+10
  settle steps, replan every 5). Rollouts on RunPod GPUs (RTX 4090 / A40) through a git-based
  job queue; per-episode records committed.
- Every phrase is rolled on the same 50 initial states (inits 0-49, seed 7), one repetition,
  so success is a percentage over 50 episodes and the resolution floor for a difference
  between two phrases on one task is ~18 pp (z / Fisher / McNemar, p<0.05).

## 2. Evidence: single-edit pairs

- A single-edit pair is two phrases on one task that differ in exactly one contiguous token
  span, both rolled at n=50 on the same inits. Built from every n=50 phrase in the bank
  (scripts/build_libero_single_edit_pairs.py -> single_edit_pairs.parquet).
- Full set: 426 pairs on 68 tasks (239 on in-finetune tasks, 187 on libero_90); 115 pairs
  are significant swings (|delta| >= 18 pp, p < 0.05); 311 are nulls, kept by design.
  Edit types: 386 replace, 26 insert, 14 delete. Categories: object/place noun, verb or
  sentence frame, preposition/particle, colour word, words added/removed, determiner,
  capitalisation, punctuation.
- Distillation evidence = the pairs with every sealed task removed (task-level filtering,
  nothing else): 369 pairs, 104 significant, flattened to 271 phrases on 58 tasks
  (26 in-finetune, 32 libero_90), columns task / category / phrase / success % / n.
  (distill_evidence_v2.csv). The canonical (fine-tuning) string of each non-sealed task
  appears as one of its rows.

## 3. Sealed test set and leakage control

- Draw (scripts/draw_libero_sealed_v2.py, seed 20260914): one seeded permutation per pool
  (in-finetune pool = 40 tasks minus libero_10/5, which had 855 probe episodes; out pool =
  all 90 libero_90 tasks); walk each permutation and accept the first 10 tasks that are not
  FLOOR tasks (0/20 canonical successes on inits 0-19). Floor status came from existing
  canonical episodes under that exact protocol or from a 320-episode canonical screen on
  the permutation-prefix tasks without data; the order was never re-drawn.
- One post-draw skip: libero_goal/7 held 103 of the 426 pairs (24%, the stove ladder) and
  was replaced by the next task in the recorded order (libero_10/4). Recorded in the draw
  file and PREREG before any test phrase existed.
- Sealed tasks. In-finetune (10): spatial/2, /3, /6, /7; goal/0, /1, /6; object/1, /6;
  libero_10/4. Out-of-finetune (10): 90/60, /70, /46, /68, /54, /7, /59, /56, /43, /35.
  Canonical screen (successes of 20) for the out half: 8, 12, 20, 20, 15, 1, 5, 18, 1, 2;
  four of ten are near-floor (<=5/20) and were kept per the rule.
- Disclosed, not filtered: sealed goal/7 shares its canonical string verbatim with
  libero_90/44 and sealed object/6 with libero_90/51; those non-sealed tasks stay in the
  evidence. Sealed in-finetune tasks share scene and template with training tasks, so the
  in-finetune result is within-scene-family transfer.
- Pre-registration: the design (draw rule, evidence, prompt, appliers, arms, n, analysis) was
  written into results/analysis/pi05_bank/PREREG.md before any test phrase, trace, rulebook
  or rollout existed; every later change (v2 prompt for draws 2-3, draw 4) is a dated
  amendment written before the corresponding generation step. Any script that touches sealed
  phrases requires FINAL_EVAL=1 and preflight-prints every phrase.
- Isolation of the distiller: each draw is a fresh, stateless multi-agent run whose agents
  receive only the evidence file (sealed tasks removed), the distill prompt and an
  instruction that they know nothing about which tasks are held out. Audit of the agent
  transcripts of draws 1-3: no agent read the sealed draw, the test phrases, the applies,
  the rollout results or the pre-registration. One deviation: the draw-2 synthesizer read
  the draw-1 rulebook (a format check; not its scores). Draw 3 read only its own output.
  From draw 4 on, the workflow prompt forbids opening any file other than the evidence, the
  prompt and the output.
- Isolation of the applier: the applier sees only the numbered rules (rationale stripped),
  the incoming instruction and a scene description; the scene description is generated from
  the incoming phrase and the start-frame image, never from the canonical (a leak gate
  regenerates any trace containing the canonical string).

## 4. Test phrases and traces

- 10 natural rephrasings per sealed task = 200 test phrases. Author: gemini-pro-latest,
  temperature 1.0, conditioned on the task's init-0 agent-view image, using the NATURAL
  block of the paper's rephrase prompt only, plus 12 human phrases from the WidowX A39 study
  (4 adult / 4 kid / 4 robot register, fixed seed) shown as style examples and labelled as
  written for a different robot. No semantic judge of any kind; only dedup and dropping the
  exact canonical. (scripts/gen_libero_naturals_v2.py, naturals_v2.parquet)
- One scene trace per (task, phrase): gemini-3.5-flash, temperature 0.4, image + phrase,
  CoVer template. (scripts/gen_libero_traces_v2.py)

## 5. Distillation

- Distiller: a multi-agent workflow (scripts/distill_libero_v2_onavg_workflow.js): 5 lens
  analysts (nouns, verbs/frames, modifiers, prepositions, cross-task consistency) each
  propose <= 6 candidate rules from the evidence; 3 adversarial refuters per candidate
  (counterexample hunter, generality checker, harm/ambiguity checker; a candidate survives
  with < 2 refutations); one synthesizer writes the 10-20-rule book under the distill
  prompt; 2 auditors (grounding, applier-executability); one revision pass. ~93-99 agents
  per draw.
- Distill prompt v1 (prompts/distill_minimal.md, frozen at tag icra2027): "Write a rulebook
  of 10-20 numbered rules for rewriting any incoming task instruction so that the robot is
  most likely to succeed. Ground every rule in consistent score differences across tasks,
  not single examples, and keep rules general enough to apply to instructions and objects
  not in the evidence. Output only the numbered rules."
- Distill prompt v2 (prompts/distill_minimal_v2.md), one clause changed: "Judge each rule by
  its expected effect on average over the tasks where it would apply: a rule is worth
  including when the score gains it produces outweigh the losses, in size and in frequency,
  even if it loses on some tasks. Include rules that change wording, not only rules that
  forbid changes, and keep rules general enough to apply to instructions and objects not in
  the evidence. Output only the numbered rules."
  Motivation: the v1 criterion (consistent across tasks) produced a book of prohibitions with
  three licensed edits that changed 56/200 test phrases, almost all onto->on / into->in.
- Draws: draw 1 = v1 prompt, Claude Opus 5, 15 rules. Draw 2 = v2, Claude Fable 5.1, 17
  rules. Draw 3 = v2, Claude Fable 5.1, 12 rules. Draw 4 = v2, Claude Opus 5, 14 rules.
  Draws 2-4 were distilled only after round 1 had been rolled and analysed end to end. Draws
  1-4 are not identical-prompt/identical-model replicates: report draw 1 as the v1 book,
  draws 2-3 as v2/Fable, draw 4 as v2/Opus.

## 6. Appliers and arms

- Applier for every rolled arm: gemini-pro-latest, temperature 1.0, thinking budget 16384,
  one greedy application per (book, phrase), prompt = prompts/rules_loop/apply.md with the
  rules, the trace and the phrase. (A Claude applier arm was started once by mistake and
  never rolled; it is not part of the results.)
- Arms: no-rephraser (the natural phrase as written), scaffold (the paper's one-rule
  scaffold book through the same applier), rulebook draws 1-4. Identical strings roll once
  and are attributed to arms at analysis time.
- Budget: 200 bases x 50 inits per arm. Rounds 1-3: 144 legs, ~28k episodes, zero lost.
  Round 4: 32 legs, zero lost (176 legs total).

## 7. Analysis

Per arm, base-weighted mean success pooled and split in-finetune / out-of-finetune; paired-
by-base sign-flip permutation test (10k permutations) of each arm vs the no-rephraser arm
and vs the scaffold arm. No pooled headline across halves is claimed.

## 8. Results (rounds 1-4; success %, n = 200 bases x 50 inits per arm)

  arm                 pooled   in    out   d_in (p)       d_out (p)      d_pooled (p)   d_vs_scaffold (p)  changed
  no rephraser         75.1   93.6  56.7   -              -              -              -                  0.00
  scaffold             76.0   93.6  58.3   +0.1 (0.87)    +1.6 (0.006)   +0.8 (0.015)   -                  0.61
  draw 1 (v1, Opus)    75.1   94.0  56.2   +0.5 (0.11)    -0.5 (0.06)    -0.0 (0.96)    -0.9 (0.033)       0.28
  draw 2 (v2, Fable)   77.3   97.6  57.0   +4.1 (0.005)   +0.3 (0.63)    +2.2 (0.009)   +1.4 (0.18)        0.91
  draw 3 (v2, Fable)   77.2   97.9  56.6   +4.3 (0.004)   -0.1 (0.91)    +2.1 (0.024)   +1.3 (0.22)        0.92
  draw 4 (v2, Opus)    77.0   97.9  56.2   +4.3 (0.004)   -0.5 (0.47)    +1.9 (0.039)   +1.1 (0.31)        0.96*
  * fraction of rolled strings differing from the base; draw 4 lowercases and strips
    punctuation, so most changes are cosmetic.

Where the in-finetune gain comes from: the two sealed tasks whose naturals used the flagged
nouns. goal/1 ("put the bowl on the stove"): 73 -> 99 under draws 2 and 3, driven by
hot plate / burner / hotplate -> stove (+47.5 / +43.3 pp over the 8-9 phrases that used
them). spatial/7: 76 -> 92 / 95 (dish -> plate, draw 3 +37.3 over 3 phrases). Every other
in-finetune sealed task was already at 95-100 with the natural phrase. Draw 4 lands on the
same two tasks (goal/1 73 -> 99, spatial/7 76 -> 95). Out-of-finetune, none of the three v2
books differs from the no-rephraser arm or from the scaffold. Three independent v2 draws,
under two distiller models, converge on the same rules and the same effect.

## 9. Timing

- Distillation: draws 1-3 ~44 min each; draw 4 30 min (99 agents, 4.9M subagent tokens).
- Apply, 200 phrases, 8 parallel workers: Gemini ~6 min (~14 s per call from wall; direct
  sequential probe 16.8 s mean, range 4.9-44.7, ~1600 thinking tokens used). With thinking
  budget 1024: 8.8 s mean, range 4.7-12.2 (probe of 6 calls only).
- Traces, 200: ~2 min at 4 workers; direct sequential 5.6 s per call.
- Rollouts: one bank-eval process ~12 s/episode; 6 init-split sub-shards per leg on a
  4090 give ~6.5 s/episode per pod; a 200-phrase arm on a 7-10-pod fleet takes 1-1.5 h.
- Draw to results, one book: ~2 h (draws 2, 3: 2 h 06 / 1 h 40).

## 10. Files

PREREG: results/analysis/pi05_bank/PREREG.md. Draw: sealed_v2_draw.json. Evidence:
single_edit_pairs.parquet, single_edit_pairs_full.{csv,md}, distill_evidence_v2.{csv,md}.
Prompts: prompts/distill_minimal.md, prompts/distill_minimal_v2.md, prompts/rules_loop/apply.md.
Books: results/analysis/pi05_bank/rulebooks/v2_draw{1,2,3,4}.md. Applies: eval_applies_v2/.
Test phrases: naturals_v2.parquet (+ naturals_v2_preflight.txt). Traces:
results/phrase_artifacts/traces_libero_v2.parquet. Rollouts: results/rules_runs/p_v2/jobs.
Cells: libero_v2_round1_cells.json. Charts: results/charts/libero_v2_round4.png (all six
arms), results/charts/single_edit_pairs.png.
