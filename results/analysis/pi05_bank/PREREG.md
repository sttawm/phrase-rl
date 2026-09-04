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
