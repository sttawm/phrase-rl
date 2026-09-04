---
name: pod-git-fills-with-checkpoint-archives
description: "committed checkpoint archives fill every pod's .git via routine pulls — symptom is a worker spinning silently on stale code"
metadata: 
  node_type: memory
  type: project
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-08-08T14:15:48.777Z
---

The trainer pushes multi-GB v11 checkpoint archives to `main` every ~2.5h (see
[[checkpoint-archive-discipline]]), and **every pod that pulls downloads them
into `.git`**. On e1 that took `.git` from 28G to 71G in a few hours and hit the
RunPod volume quota.

**The symptom is not a disk error.** `git pull` fails with "Disk quota exceeded",
and `rules_loop_worker.sh` swallows it (`2>/dev/null`) then falls back to
`git reset --hard origin/main` — a ref that can never advance without a
successful fetch. The worker spins forever on stale code: process alive, log
frozen at its startup line, jobs never picked up, run dir for the current run
absent from the pod entirely. It looks like a hung worker, not a full disk.
`git gc` does not help: the archive blobs are reachable from history.

**How to apply:** when a pod worker is alive but idle, check
`du -sh /workspace/phrase-rl/.git` and run a bare `git pull` to see the real
error before debugging the worker. The durable fix is to replace the pod's clone
with a partial one — verify `git log origin/main..HEAD` is empty and
`git status --porcelain` is clean (excluding gitignored payloads) first, then
drop `.git`, re-clone with `--filter=blob:none`, and move `data/` and
`.venv-gen/` (~14G together) into the new tree. Keep the remote URL by piping it
host-to-host; **never run `pgrep -af`/`ps` on a command line containing it** —
the URL embeds a GitHub token and that is how one got printed. Related:
[[pod-untracked-collision-wedge]], [[git-gc-aggressive-interrupt]].

**Sparse-checkout blindness (2026-08-12):** the flip side of the partial-clone
fix — e6's clone sparse-excludes `results/checkpoints/archive/`, so the v12
eval loop saw an empty archive dir and idled ("nothing unclaimed") while
origin held verified checkpoints. Diagnosis tell: `git log` HAS the ckpt
commit but `ls` shows no dir. Fix: exclude by `archive/*` (contents) and
re-include the current run's pattern —
`git sparse-checkout set --no-cone "/*" "!/results/checkpoints/archive/*" "/results/checkpoints/archive/v12_step_*"`
(excluding the DIR itself makes re-inclusion impossible per gitignore rules).
