---
name: runpod-podstate-backup
description: "RunPod stops wipe the container disk — bashrc tokens, ssh keys, tmux; back them up to /workspace/.podstate + restore.sh before any stop"
metadata: 
  node_type: memory
  type: project
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
---

RunPod "stop pod" preserves only /workspace; the container disk (/root, apt packages) is wiped. Anything session-critical outside /workspace dies: `~/.bashrc` (HF/GEMINI tokens), `~/.ssh` (inter-pod transfer keys), apt-installed tmux.

**Why:** a stopped-then-started pod that lost its tokens/keys silently breaks trainers (keyless HF pulls), pod-to-pod scp, and tmux relaunches — each a confusing failure hours later.

**How to apply:** before a planned stop (and routinely on new pods), run the backup: copy `~/.bashrc` and `~/.ssh` into `/workspace/.podstate/` and write `/workspace/restore.sh` (restores both + reinstalls tmux). After start: `bash /workspace/restore.sh`, then relaunch via the repo's run script for that pod (trainers resume from latest checkpoint). Set up on all phrase-rl pods 2026-07-10. Related: [[runpod-vulkan-rendering]].
