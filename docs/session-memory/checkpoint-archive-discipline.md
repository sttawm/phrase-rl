---
name: checkpoint-archive-discipline
description: ALWAYS archive checkpoints off-pod immediately (user order 2026-07-22); repo mirror + split-tar git archive; tmp scratchpad is a loss zone
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-07-22T23:02:46.988Z
---

User order (2026-07-22, emphatic): "Remember we have to ALWAYS save checkpoints! Always! As per CLAUDE.md. Keep them somewhere! Git is fine!"

**Why:** The phase-2 flow/L2-era adapters were nearly lost (pods retired before archiving; they only survived by accident in the gitignored local `results/checkpoints/`). A dozen `A_step_*`/`B_step_*` dirs in the session-scratchpad archive were later found 0 bytes = permanently lost, because `/private/tmp` scratchpads are session-scoped and wipeable.

**How to apply:**
- The moment a training run produces `best_val`/`final` (or a named RL step gets deployed), rsync it to local `results/checkpoints/<run>/` AND pack into git with `scripts/ckpt_archive.sh pack <dir>` (90MB split tars under `results/checkpoints/archive/`, sha256 manifest; `unpack` restores — also the transport path to new pods via git pull).
- Never store archives under the session scratchpad `/private/tmp/...` — repo-local dirs only.
- Before ANY pod retirement: scan `results/checkpoints/` + `/workspace/adapters` on the pod and pull everything missing (see [[gitignored-data-payload]] for the data/ analog).
- Local `results/checkpoints/pod1_mirror/` + `tmp_archive_rescue/` hold the full uncurated mirror (gitignored); curated named checkpoints go to the git archive.
