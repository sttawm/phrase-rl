---
name: pod-untracked-collision-wedge
description: "scp'ing tracked files to pods before committing locally causes untracked-vs-tracked collisions that block every rebase; repair recipe + prevention"
metadata: 
  node_type: memory
  type: project
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-07-30T19:58:45.879Z
---

The recurring pod git wedge ("could not detach HEAD", perpetual PUSH-DEFERRED in ckpt-sync loops) is caused by scp'ing files to a pod BEFORE their local commit reaches origin: when the commit later lands, the pod copy is an untracked file that `git pull --rebase` refuses to overwrite, and every sync loop's commit+pull+push chain fails silently from then on.

**Why:** ckpt-sync loops swallow stderr, so the archive backlog grows invisibly (v7a 320/340 + v7b adapters sat unpushed on L40S; discovered 2026-07-28 only via `git log origin/main..HEAD`).

**How to apply:** Repair = for each collision path: `git show origin/main:$p | cmp -s - $p || cp $p /workspace/podlocal_backup/`; `rm $p`; then `git pull --rebase`; resolve jsonl conflicts by union-dedup on step; chunked push oldest-first (`git push origin HEAD~N:main`) to stay under GitHub's ~2GB pack limit. Replacing scripts mid-run is safe (running bash keeps its inode). Prevention = commit+push locally FIRST, pull on pod; never bare-scp tracked files. Check `git log origin/main..HEAD | wc -l` on every pod status pass alongside [[checkpoint-sync-liveness]].

**AUTOSTASH-CLOBBER variant (local side, 2026-07-30):** scp'ing a pod's copy of a tracked results file ONTO the local checkout is the mirror-image footgun: the stale copy dirties the tree, rides `pull --rebase --autostash` back on top of the freshly-pulled newer version, and the next blanket `git add -A` commit DELETES the newer rows from origin (lost the measured v7f pol-80 point; restored from the pod's commit). Prevention = scp remote copies only into untracked gitignored staging paths (`results/analysis/_scp_*`), merge at render time (max-n dedup), and diff-check any `git add -A` commit that follows a pull-with-autostash for unexpected deletions (`git show --stat` shows the `1 -`).
