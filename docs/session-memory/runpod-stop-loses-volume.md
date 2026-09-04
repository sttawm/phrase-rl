---
name: runpod-stop-loses-volume
description: "RunPod stop can wipe the /workspace VOLUME on some pods, not just the container disk — budget a full re-bootstrap, not a restore"
metadata: 
  node_type: memory
  type: project
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-09-04T08:17:33.377Z
---

Stopping a pod is documented (and previously assumed here) to wipe only the
container disk, with `/workspace` persisting. On 2026-09-04, after parking 14
render pods overnight and restarting them, that held for some and not others:

- **rv8, rv16**: `/workspace` intact, 28G, repo + INT-ACT present → `restore.sh`
  was enough, worker up in a minute.
- **rv9, rv10**: `/workspace` came back at **76K of 80G** — no repo, no INT-ACT,
  only the small files written at park time. Effectively a fresh volume.

Pattern (weak, n small): the wiped ones were created on **COMMUNITY** cloud;
the survivors were an original pod and one recreated on **SECURE**. Not proven,
but worth assuming community-cloud pods lose the volume.

**Why it burned us:** the overnight park saved ~$100 but cost a ~45-min
scripted re-bootstrap per lost pod, discovered only when workers failed with
`bash: scripts/rules_loop_worker.sh: No such file or directory` — which looks
like a launch bug, not data loss. (The tell that it IS data loss: the launcher's
`cd /workspace/phrase-rl &&` silently short-circuits and the nohup then runs
from `$HOME`.)

**How to apply:** before parking, decide per pod whether a re-bootstrap is
acceptable; treat stop as closer to terminate for community-cloud pods. After
any restart, verify with `du -sh /workspace` and a `-d /workspace/phrase-rl/.git`
check BEFORE launching workers — a worker launched into an empty volume claims
jobs it cannot run. Endpoints also change on restart: refresh ip/port in
`lb_ssh.json` first. Related: [[runpod-podstate-backup]],
[[base-volume-reset-checklist]], [[pod-relaunch-process-verification]].
