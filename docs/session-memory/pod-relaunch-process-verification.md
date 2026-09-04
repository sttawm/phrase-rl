---
name: pod-relaunch-process-verification
description: "After any pod relaunch, verify the PROCESS (pgrep + log mtime + progress line), never tmux session existence — empty-session-alive is the dangling-venv signature"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-08-04T18:45:19.719Z
---

A relaunch is not verified until `pgrep` finds the process AND the log mtime is fresh AND one new progress line has appeared. tmux session existence proves nothing: after a migration wipes /root, dangling venv symlinks make the session's command die instantly ("bash: line 1 ... No such file or directory" in the log) while the session persists as an empty shell.

**Why:** v9 sat dead ~35h (2026-08-02/03) because the recovery pass reported "V9 LAUNCHED" off `tmux ls` alone; all three sessions (train/score/v9sync) were corpses. Compounds with [[pod-clone-venv-symlink-audit]] (the dangling links) and [[checkpoint-sync-liveness]] (the mirror died the same way).

**How to apply:** every relaunch ends with: `pgrep -cf <proc>` ≥1, log mtime < 5 min, and after the model-load window one substantive log line (e.g. "resumed at step N"). Re-arm the liveness watchdog in the same pass — recovery isn't done until the watchdog is armed.

**Deploy-at-restart variant (2026-08-04):** a push is not a deploy. The launch script's internal `git pull || true` swallows failures against a flaky GitHub link, silently relaunching the OLD config (v9c attempt 1 restarted with stale 16/16 flags). Before arming a config-change restarter: verify the ON-DISK script on the pod shows the new values. After the restart: verify the flags on the LIVE cmdline (`pgrep -af | grep -o -- --flag`), never just the launch banner.
