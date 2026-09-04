---
name: base-volume-reset-checklist
description: "a RunPod base-volume reset wipes /root — venvs (symlinked), tmux, git-lfs, git identity; the LFS hook then fails closed and blocks all pushes"
metadata: 
  node_type: memory
  type: project
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-08-08T17:33:46.261Z
---

Resizing a RunPod volume resets the base volume (`/root` and apt packages).
`/workspace` survives. What actually breaks, in the order it bites:

1. **Venvs, on the TRAIN pod only.** `bootstrap_trainpod.sh` creates them at
   `/root/venv-pi0` and `/root/venv-gen` and symlinks `.venv` / `.venv-gen` to
   them — "container-disk venvs (MFS-safe)". After a reset the symlinks survive
   and the targets do not, so the dirs exist with no `bin/python`. Rebuild with
   `SKIP_LAUNCH=1 bash scripts/bootstrap_trainpod.sh A` (~10 min), then launch
   the arm separately. Eval pods are unaffected: their venvs are real
   directories on `/workspace`. Pointing the train pod's venvs at `/workspace`
   would make resets harmless. See [[pod-clone-venv-symlink-audit]].
2. **tmux** is gone — every launcher dies with `tmux: command not found`.
   `apt-get install -y tmux`.
3. **git-lfs** is gone but `.git/hooks/pre-push` remains, and it FAILS CLOSED.
   Every push dies with a message about LFS in a repo that has no
   `.gitattributes` and has never used LFS. Delete the four hooks
   (`pre-push post-checkout post-commit post-merge`). This silently stopped a
   checkpoint archive from reaching origin four times before I read the full
   error instead of the summary.
4. **git identity**, if it lived in `.git/config` rather than `~/.gitconfig` —
   also lost by re-cloning a pod repo. Commits fail with "Author identity
   unknown", which the pusher swallows. Re-set `user.name` / `user.email`.

**Why:** none of these announce themselves as a reset consequence; each looks
like an unrelated bug at the point of failure.

**How to apply:** after any resize or reset, run through 1–4 before restarting
work, and verify with `.venv-gen/bin/python -c "import phrase_rl.phase2_train"`,
`tmux -V`, `ls .git/hooks | grep -v sample`, and `git config user.name`.
Related: [[pod-git-fills-with-checkpoint-archives]],
[[pod-relaunch-process-verification]].
