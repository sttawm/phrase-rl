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
