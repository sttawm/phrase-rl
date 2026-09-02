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
