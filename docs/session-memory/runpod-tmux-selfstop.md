---
name: runpod-tmux-selfstop
description: "RunPod self-stop from tmux — nohup'd background stop dies with tmux server teardown; run runpodctl stop in the foreground"
metadata: 
  node_type: memory
  type: project
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
---

Pod self-stop via `nohup sh -c "sleep 20; runpodctl stop pod $ID" &` at the end
of a tmux-run script fails SILENTLY when that script's session is the last one:
bash exits → pane closes → tmux SERVER shuts down and kills the whole process
tree (nohup only shields SIGHUP, not the server's kill). Cost ~15h idle GPU on
shard2 (2026-07-15, twice).

**Why:** tmux server teardown, not SIGHUP.

**How to apply:** put `sleep 20; runpodctl stop pod "$RUNPOD_POD_ID"` in the
FOREGROUND as the script's last lines (fixed in scripts/run_rollout_shard.sh and
run_verifier_data.sh). Note `runpodctl config --apiKey` prints an "Unauthorized"
SSH-key error with pod-scoped keys — harmless, the stop still works. Related:
[[runpod-podstate-backup]], [[pkill-self-match-footgun]].

Confirmed-safe patterns (2026-07-18 fleet wind-down):
- A watcher daemon `nohup`'d from a PLAIN SSH command (never inside tmux)
  reparents to init and survives both ssh disconnect and tmux teardown —
  fine for wait-for-marker → `runpodctl remove` self-destruct daemons.
- RUNPOD_POD_ID / RUNPOD_API_KEY are absent in ssh login shells; source them
  from /proc/1/environ: `export $(tr '\0' '\n' < /proc/1/environ | grep -E
  '^RUNPOD_(POD_ID|API_KEY)=' | xargs)`.
- `remove` (pod + /workspace gone) for disposable pods whose results are in
  git — gate on a results-pushed/archive-verified check; `stop` + podstate
  backup only for pods meant to be revived.
