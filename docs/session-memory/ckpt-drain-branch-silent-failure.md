---
name: ckpt-drain-branch-silent-failure
description: "Side-branch checkpoint drain reported progress but landed 0 archives; verify with git ls-tree on the branch, never with drain-log line counts"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-08-07T04:11:18.085Z
---

2026-08-06: to escape slow main-branch pushes, checkpoint archives were drained
to a dedicated `ckpt-l40s` branch by a loop that logged one line per pushed
commit. The log grew to 33 lines — but `git ls-tree -r origin/ckpt-l40s
results/checkpoints/archive/` returned **0 archive files**, and origin/main held
only 3 of 25 v10 steps. The archives existed solely in the L40S's local git
objects and on pod6's filesystem: two single copies, both on machines slated for
shutdown.

Root cause of the underlying push failures: the v10 sync loop used
`timeout 200` per push attempt for ~400MB archive commits — far too short, so
every attempt died mid-transfer and the loop moved on.

**Why:** drain/sync loops report *attempts*, not *arrivals*. A log line means the
command returned, not that objects landed on the remote.

**How to apply:** verify archive durability only by listing the remote tree
(`git ls-tree -r --name-only origin/<branch> results/checkpoints/archive/ | grep
-c <run>_step`) and compare against the trained-step count. Give archive pushes
generous per-attempt timeouts (1800s+, 6 retries) — see the v11 pattern in
`/workspace/v11sync.sh`. Never terminate (only stop) a pod holding a
single-copy archive; a stopped pod keeps its volume. Related:
[[checkpoint-sync-liveness]], [[checkpoint-archive-discipline]].
