---
name: git-gc-aggressive-interrupt
description: git gc --aggressive on a huge repo writes its new pack early — kill it once .git shrinks rather than letting it block GPU work for hours
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-08-08T05:44:49.739Z
---

`git gc --prune=now --aggressive` on this repo's multi-tens-of-GB `.git` (it
carries committed checkpoint archives, see [[checkpoint-archive-discipline]])
runs for hours, and while it runs `.git` GROWS — the new pack is written before
the old objects are dropped. On e1 it went 65G → 71G at the 70-minute mark.

The delta-recompression tail is where the hours go, and it is worth far less
than the pod's GPU time. Killing the gc after the new pack has landed captures
almost all of the benefit: `.git` 71G → 26G, `/workspace` 111G → 72G, and
`git fsck --connectivity-only` reported only dangling commits (harmless — the
reflog expire had already run).

**Why:** an aggressive repack blocks anything that pulls (the index lock), so a
scoring or training job chained behind it idles the GPU for the whole run.

**How to apply:** don't chain GPU work behind an aggressive gc. If one is
already running and blocking, check `du -sh .git` — if it has dropped below the
starting size, kill it, then `rm -f .git/objects/pack/tmp_pack_* .git/gc.log
.git/gc.pid` and `git fsck --connectivity-only` to confirm. Prefer plain
`git gc` (no `--aggressive`) when the goal is just reclaiming space. Related:
[[pod-untracked-collision-wedge]].
