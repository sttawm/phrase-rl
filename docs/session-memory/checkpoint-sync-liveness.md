---
name: checkpoint-sync-liveness
description: "v7 lost snapshots 60-280 because the mirror died silently at step 40 — archive watchers must be liveness-checked at every status pass, not just at setup"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-07-25T15:30:08.480Z
---

Setting up a checkpoint mirror/archive once is NOT the discipline — verifying it
is still moving is. v7's L40S mirror synced steps 20+40 then died silently;
training ran 2+ days; the pod was terminated. Weights were saved ONLY by luck:
/workspace was a network volume that survived termination (EXPERIMENT.md
2026-07-25 loss record + correction). Also: mfs network volumes exhibit
cold-mount metadata lag right after pod boot — early probes can read empty dirs
/ "not a git repository" for paths that are fine minutes later; re-probe before
declaring data lost. Related: [[checkpoint-archive-discipline]].

**Why:** a dead sync looks identical to a healthy one from the outside — the only
signal is the newest archived step falling behind the newest trained step.

**How to apply:** every training run gets a sync watcher that (1) pushes each
snapshot to the git archive ON CREATION (ckpt_archive.sh), (2) round-trip
verifies (list archive, compare to latest step), and (3) is checked for
liveness — newest-archived vs newest-trained — at EVERY fleet status pass and
before ANY pod shutdown/retirement answer. "Safe to shut down" requires
checking archive freshness for training pods, not just git cleanliness.
