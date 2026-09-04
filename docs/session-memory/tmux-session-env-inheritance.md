---
name: tmux-session-env-inheritance
description: "New tmux sessions inherit the tmux SERVER's original env, not the launching shell's exports — source secrets inside the tmux command string"
metadata: 
  node_type: memory
  type: reference
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-08-31T23:12:06.455Z
---

`tmux new-session -d '<cmd>'` runs `<cmd>` with the environment the tmux **server** was first started with — the launching shell's exports (HF_TOKEN etc.) do NOT carry over. On 2026-08-31 the rules-loop score server booted this way without HF_TOKEN and died on the gated PaliGemma repo while the token sat in ~/.bashrc the whole time.

**How to apply:** any pod service launched via tmux must eval its secrets *inside* the command string: `tmux new-session -d "eval \"\$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc)\" && <cmd>"`. Same trap for git identity: a worker relaunched in a fresh tmux lost `user.email` that the original session carried in env — put identity in repo-local git config, never rely on session env. Related: [[runpod-tmux-selfstop]], [[pod-relaunch-process-verification]].
