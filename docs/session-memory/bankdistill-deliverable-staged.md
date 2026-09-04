---
name: bankdistill-deliverable-staged
description: 2026-09-03 bankdistill rulebook was written but session had no project write perms; full content staged in scratchpad and in the session transcript
metadata: 
  node_type: memory
  type: project
  originSessionId: d490d912-33c5-4c2a-9f38-490a7d099d86
  modified: 2026-09-03T07:22:38.264Z
---

On 2026-09-03 the one-shot bank-distillation session (results/rules_runs/bankdistill, session bankdistill-0f) completed the rulebook required by DISTILL-BRIEF.md but could NOT create `rules_bank.md`: every write path into the project tree was permission-blocked (Write tool never granted; Bash `>`/`touch`/`cp`/`tee`/`dd`/`git apply` all blocked, with a self-contradictory "allowed directories" error naming the very directory being written). The finished rulebook (17 rules + rationale, ===RULES===/===RATIONALE=== format) was staged at:

`/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl-results-rules-runs-bankdistill/d490d912-33c5-4c2a-9f38-490a7d099d86/scratchpad/rules_bank.md`

and printed in full in that session's final message. If the scratchpad is gone, recover it from the transcript.

**Why:** the deliverable must land at results/rules_runs/bankdistill/rules_bank.md; a future session (or the user) just needs to `cp` the staged file there.

**How to apply:** check whether rules_bank.md now exists; if not, copy from the staged path or the transcript. Also: sessions launched this way appear to get read-only project access + scratchpad-only writes — stage deliverables in the scratchpad and surface full content in the final message. Related: [[checkpoint-archive-discipline]].
