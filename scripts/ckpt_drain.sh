#!/bin/bash
# Training-pod checkpoint drain: pack each phase2_${RUN}/step_XXXX to git as
# ${RUN}_step_XXXX split-tars, push, and VERIFY ON ORIGIN before counting it
# drained. Standing discipline: a push exit code is not an arrival — git push
# exits 0 when nothing was staged (that is how v11 steps 0140/0150 were logged
# "pushed" while origin held 0130). Local step dirs are pruned only after
# origin verification, keeping the newest KEEP locally.
#   RUN=v12 setsid nohup bash scripts/ckpt_drain.sh > /workspace/drain_v12.log 2>&1 &
set -uo pipefail
RUN="${RUN:?set RUN, e.g. v12}"
KEEP="${KEEP:-3}"
CKDIR="results/checkpoints/phase2_${RUN}"
cd /workspace/phrase-rl
mark() { echo "[drain-$RUN $(date -u +%H:%M)] $*"; }

while true; do
  for d in $(ls -d ${CKDIR}/step_* 2>/dev/null | sort); do
    s=$(basename "$d" | grep -oE "[0-9]+$")
    name="${RUN}_step_${s}"
    if git ls-tree --name-only origin/main -- "results/checkpoints/archive/${name}.manifest.json" | grep -q .; then
      continue  # already on origin
    fi
    # step dir may still be mid-copy from save_latest; require the safetensors
    [ -f "$d/adapter_model.safetensors" ] || continue
    mark "packing $name"
    stage=$(mktemp -d)
    cp -r "$d" "$stage/$name" || { rm -rf "$stage"; continue; }
    bash scripts/ckpt_archive.sh pack "$stage/$name" || { mark "PACK FAIL $name"; rm -rf "$stage"; continue; }
    rm -rf "$stage"
    git add "results/checkpoints/archive/${name}."*
    git commit -q -m "ckpt archive: $name" || true
    for i in 1 2 3; do
      timeout 1800 bash -c "git -c rebase.autoStash=true pull -q --rebase origin main && git push -q origin HEAD:main" && break
      git rebase --abort 2>/dev/null; sleep 30
    done
    git fetch -q origin main
    n_local=$(ls results/checkpoints/archive/${name}.tgz.part-* 2>/dev/null | wc -l | tr -d ' ')
    n_origin=$(git ls-tree --name-only origin/main -- results/checkpoints/archive/ | grep -c "^.*${name}\.tgz\.part-" || true)
    if [ "$n_local" -gt 0 ] && [ "$n_local" = "$n_origin" ]; then
      mark "VERIFIED on origin: $name ($n_origin parts)"
    else
      mark "NOT-ON-ORIGIN: $name (local $n_local vs origin $n_origin) -- retrying next pass"
    fi
  done
  # prune local step dirs that are origin-verified, keeping the newest KEEP
  verified=""
  for d in $(ls -d ${CKDIR}/step_* 2>/dev/null | sort | head -n -${KEEP}); do
    s=$(basename "$d" | grep -oE "[0-9]+$"); name="${RUN}_step_${s}"
    parts=$(git ls-tree --name-only origin/main -- results/checkpoints/archive/ | grep -c "${name}\.tgz\.part-" || true)
    [ "$parts" -gt 0 ] && { rm -rf "$d"; verified="$verified $s"; }
  done
  [ -n "$verified" ] && mark "pruned local:$verified"
  sleep 300
done
