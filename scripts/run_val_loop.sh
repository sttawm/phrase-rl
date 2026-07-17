#!/usr/bin/env bash
# ROLLING VAL LOOP v2 (pod 3): screen the newest relayed checkpoint per arm on
# RED-TEAM inputs. TUNED ARM ONLY (576 eps: 8 tasks x 24 layouts x 3 reps) —
# baselines come from the battery at x12. Hardened: VRAM guard before Qwen gen,
# worker exit codes checked, row-count verified, failures RETRY (cap 2) instead
# of silently skipping, stage markers flushed.
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl

mark() { echo "[val-loop $(date +%H:%M:%S)] $*" | tee -a /workspace/valloop_stages.log; }

vram_wait() {  # wait until >= $1 MiB free
  while true; do
    free=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) ))
    [ "$free" -ge "$1" ] && return 0
    mark "waiting for VRAM ($free MiB free, need $1)"; sleep 20
  done
}

pick_spread_ready() {  # farthest-point sampling over training steps: keep the
  # screened set SPREAD over the run instead of chasing only the newest ckpt
  local arm=$1 cands="" done_steps="" stamp step tries
  for d in /workspace/adapters/$arm/*/; do
    [ -f "$d/READY" ] || continue
    stamp=$(basename "$d")
    case "$stamp" in
      step_*) step=$((10#${stamp#step_})) ;;
      *)      step=60 ;;   # legacy epoch stamps = the ~s60 relays
    esac
    if [ -f "results/val_screens/.done_${arm}_${stamp}" ]; then
      done_steps="$done_steps $step"
    else
      tries=$(cat "results/val_screens/.try_${arm}_${stamp}" 2>/dev/null || echo 0)
      [ "$tries" -ge 2 ] && continue
      cands="$cands $stamp:$step"
    fi
  done
  [ -z "$cands" ] && return 1
  local best="" best_d=-1 best_s=-1 cs mind ds dd
  for cs in $cands; do
    stamp=${cs%%:*}; step=${cs##*:}; mind=999999
    for ds in $done_steps; do
      dd=$(( step > ds ? step - ds : ds - step ))
      [ "$dd" -lt "$mind" ] && mind=$dd
    done
    if [ "$mind" -gt "$best_d" ] || { [ "$mind" -eq "$best_d" ] && [ "$step" -gt "$best_s" ]; }; then
      best=$stamp; best_d=$mind; best_s=$step
    fi
  done
  echo "/workspace/adapters/$arm/$best/"
}

REPEATS=${VAL_REPEATS:-3}
PREFIX=$([ "$REPEATS" = 3 ] && echo screen || echo eval${REPEATS})
mkdir -p results/val_screens
while true; do
  did=0
  for ARM in ${VAL_ARMS:-A B}; do
    DIR=$(pick_spread_ready $ARM) || continue
    STAMP=$(basename "$DIR")
    T="results/val_screens/.try_${REPEATS}x_${ARM}_${STAMP}"
    echo $(( $(cat "$T" 2>/dev/null || echo 0) + 1 )) > "$T"
    mark "screen $ARM/$STAMP attempt $(cat $T)"
    vram_wait 20000
    if ! .venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
        --tasks data/val_screen_frames.parquet --ctx data/val_screen_frames.parquet \
        --assets data/val_screen_assets.parquet --adapter "$DIR" \
        --out data/screen_phrases_all.parquet >> /workspace/valloop_stages.log 2>&1; then
      mark "GEN FAILED $ARM/$STAMP (will retry, cap 2)"; continue
    fi
    NW=${VAL_WORKERS:-2}   # 4 on 46GB A40s, 2 on 24GB cards
    NW=$NW .venv-gen/bin/python - <<'PY'
import os
import pandas as pd
nw = int(os.environ["NW"])
d = pd.read_parquet("data/screen_phrases_all.parquet")
d = d[d.arm == "tuned"]                       # tuned-only: baselines live in the battery
tasks = sorted(d.task.unique())
for w in range(nw):
    wt = [t for i, t in enumerate(tasks) if i % nw == w]
    d[d.task.isin(wt)].to_parquet(f"data/screen_w{w}.parquet", index=False)
print("tuned rows:", len(d), "| workers:", nw)
PY
    rm -f data/screen_out_w*.parquet
    cd /workspace/INT-ACT
    pids=""
    for w in $(seq 0 $((NW-1))); do
      /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
        --int-act-root /workspace/INT-ACT \
        --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
        --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
        --phrases /workspace/phrase-rl/data/screen_w$w.parquet \
        --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
        --repeats $REPEATS \
        --out /workspace/phrase-rl/data/screen_out_w$w.parquet > /workspace/screen_w$w.log 2>&1 &
      pids="$pids $!"
    done
    wfail=0
    for p in $pids; do wait $p || wfail=1; done
    cd /workspace/phrase-rl
    if [ $wfail = 1 ]; then mark "WORKER FAILED $ARM/$STAMP (retry)"; continue; fi
    OUT="results/val_screens/${PREFIX}_${ARM}_${STAMP}.parquet" EXPECT=$((192 * REPEATS)) \
    .venv-gen/bin/python - <<'PY' || { mark "MERGE FAILED $ARM/$STAMP (retry)"; continue; }
import glob
import os
import pandas as pd
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/screen_out_w*.parquet"))], ignore_index=True)
assert len(d) == int(os.environ["EXPECT"]), f"expected {os.environ['EXPECT']} eps, got {len(d)}"
out = os.environ["OUT"]
d.to_parquet(out, index=False)
s = d.groupby("task").success.mean() * 100
print(f"[val-loop] RESULT {out}: pooled {d.success.mean()*100:.1f}% |", s.round(1).to_dict(), flush=True)
PY
    touch "results/val_screens/.done_${REPEATS}x_${ARM}_${STAMP}"
    git add results/val_screens/ && git -c user.name=pod3 -c user.email=pod@runpod commit -q -m "val screen $ARM $STAMP [pod]" && git -c user.name=pod3 -c user.email=pod@runpod pull -q --rebase --no-edit && git push -q
    mark "screen $ARM/$STAMP DONE"
    did=1
  done
  [ $did = 0 ] && sleep 240
done
