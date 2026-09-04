---
name: pkill-self-match-footgun
description: pkill -f <pattern> inside an ssh command kills the ssh session itself when the pattern appears in the command text — bracket the first char
metadata: 
  node_type: memory
  type: project
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-09-03T21:18:59.827Z
---

`ssh pod 'pkill -f phrase_rl.phase2_train; <more steps>'` kills its OWN session: the remote shell's command line contains the literal pattern, `pkill -f` matches it, the connection dies, and the remaining steps never run — looking exactly like a network blip (silent, no output).

**Why it burned us (2026-07-10):** ~10 consecutive "failed" v3 relaunch attempts across two pods and a jump host were all self-kills, misdiagnosed as datacenter outages for an hour.

**How to apply:** bracket the first character so the regex doesn't match its own literal text: `pkill -f "[p]hrase_rl.phase2_train"`. Or skip pkill entirely when the target is already dead (check first with a read-only command). Prefer fire-and-forget mutations: `nohup <script> &` so the pod finishes even if the connection drops. Related: [[runpod-podstate-backup]].

**Third disguise (2026-07-30): tmux session-name PREFIX matching.** `tmux has-session -t boards` matches a session named `boardsext` (tmux resolves target names by prefix) — a relay of session-waiters deadlocked in a ring because the finished session's name was a prefix of a waiting one. Fix: exact-match with `-t =name`, or never give sessions prefix-overlapping names.

**Fourth disguise (2026-09-04): the FALSE-BUSY drain check.** An unbracketed `ssh pod 'pgrep -f "rules_loop_jobs" | wc -l'` used as a "is this pod still working?" test always returns >=1, because the ssh remote command line itself contains the pattern. Effect is inverted from a self-kill and quieter: nothing dies, the drain loop just reports EVERY pod busy forever and a fleet shutdown silently never happens (14 idle GPUs kept burning). Same bracket fix; and note read-only pgrep needs the guard exactly as much as pkill does. Tell: *all* targets busy, including ones you know are idle.

**Generalization (2026-07-28, burned twice more):** the bracket must cover EVERY literal occurrence of the pattern anywhere in a process's command line, not just the pgrep/pkill argument. Two new disguises: (1) a chain-waiter `while pgrep -f "[v]7_dev8_backfill"` deadlocked on ITSELF because the same bash -c string later contained the plain literal `scripts/v7_dev8_backfill.sh` in an embedded tmux launch — the bracketed pattern matched that other occurrence; (2) `pkill -f "tar -C /workspace ..."` inside an ssh command killed the ssh session (unbracketed, pattern in own cmdline, exit 255 with zero output — looks exactly like a network failure). Rules: bracket ALWAYS, grep for the pattern text across the whole command string before running, prefer waiting on session existence (`tmux has-session`) or worker-process names over runner-script names.
