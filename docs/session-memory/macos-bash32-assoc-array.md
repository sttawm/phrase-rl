---
name: macos-bash32-assoc-array
description: "macOS bash 3.2 silently breaks `declare -A` maps in pod scripts — every lookup returns the LAST value; nearly stopped the wrong RunPod pod"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-09-02T20:14:40.685Z
---

Local scripts on this Mac run under bash 3.2 (`/bin/bash`), which has no
associative arrays. `declare -A IDS=([rv2]=x [rv3]=y)` errors but the script
CONTINUES: each `IDS[rvN]=v` collapses to `IDS[0]=v` (string index → arith → 0),
so `${IDS[$p]}` returns the last-assigned value for every key. On 2026-09-03
this made a pod-pause script call the RunPod stop API with rv4's id while
labeled "rv2" — stopped a live worker pod out of order (survived only because
it was between jobs).

**Why:** the error from `declare -A` is non-fatal without `set -e`, and the
wrong-value failure mode is silent and plausible-looking (a valid pod id).

**How to apply:** in any local script mapping names→ids, never use `declare -A`.
Use a `case` function (`pod_id() { case $1 in rv2) echo ...;; esac; }`) or
parallel arrays. Related: [[pkill-self-match-footgun]] — same class of
"remote-control script footgun that acts on the wrong target".
