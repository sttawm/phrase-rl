#!/usr/bin/env bash
# ROLLING VAL LOOP (pod 3): perpetually screen the newest relayed checkpoint of
# each arm on RED-TEAM inputs. 8 tasks x 24 layouts x 3 reps per screen (~1-1.5h).
# Adapters arrive via Mac relay in /workspace/adapters/{A,B}/<stamp>/ with a READY
# marker. Results: results/val_screens/screen_<arm>_<stamp>.parquet, pushed.
# One iteration failure never kills the loop.
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl

# ---- one-time inputs -------------------------------------------------------
if [ ! -f data/valtier_ert_assets.parquet ]; then
  .venv-gen/bin/python -m phrase_rl.gemini_redteam_assets \
    --ert results/phrase_artifacts/valtier_ert_bank.json \
    --contexts data/valtask_frames.parquet \
    --out data/valtier_ert_assets.parquet || exit 1
fi
if [ ! -f data/val_screen_assets.parquet ]; then
  .venv-gen/bin/python - <<'PY'
import pandas as pd
a = pd.read_parquet("results/phrase_artifacts/redteam_eval_assets.parquet")[["task","ert_instruction","trace"]]
b = pd.read_parquet("data/valtier_ert_assets.parquet")[["task","ert_instruction","trace"]]
pd.concat([a, b], ignore_index=True).to_parquet("data/val_screen_assets.parquet", index=False)
f1 = pd.read_parquet("data/contexts_0c_tasks.parquet")
f2 = pd.read_parquet("data/valtask_frames.parquet")
cols = [c for c in f1.columns if c in f2.columns]
pd.concat([f1[cols], f2[cols]], ignore_index=True).to_parquet("data/val_screen_frames.parquet", index=False)
print("combined assets + frames for", len(a)+len(b), "tasks")
PY
fi

pick_newest_ready() {  # $1 = arm; echoes newest READY dir not yet screened
  for d in $(ls -dt /workspace/adapters/$1/*/ 2>/dev/null); do
    [ -f "$d/READY" ] || continue
    stamp=$(basename "$d")
    [ -f "results/val_screens/.done_$1_$stamp" ] && continue
    echo "$d"; return 0
  done
  return 1
}

mkdir -p results/val_screens
while true; do
  did_work=0
  for ARM in A B; do
    DIR=$(pick_newest_ready $ARM) || continue
    STAMP=$(basename "$DIR")
    echo "[val-loop] screening arm $ARM checkpoint $STAMP"
    OUT="results/val_screens/screen_${ARM}_${STAMP}.parquet"
    # 1) generate phrases (Qwen on GPU, alone)
    .venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
      --tasks data/val_screen_frames.parquet --ctx data/val_screen_frames.parquet \
      --assets data/val_screen_assets.parquet --adapter "$DIR" \
      --out data/screen_phrases.parquet || { echo "[val-loop] GEN FAILED $ARM/$STAMP"; touch "results/val_screens/.done_${ARM}_${STAMP}"; continue; }
    # 2) rollouts: 2 workers on task halves (pi0 x2 fits after Qwen exits)
    .venv-gen/bin/python - <<'PY'
import pandas as pd
d = pd.read_parquet("data/screen_phrases.parquet")
tasks = sorted(d.task.unique())
d[d.task.isin(tasks[:4])].to_parquet("data/screen_h1.parquet", index=False)
d[d.task.isin(tasks[4:])].to_parquet("data/screen_h2.parquet", index=False)
PY
    cd /workspace/INT-ACT
    for h in 1 2; do
      /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
        --int-act-root /workspace/INT-ACT \
        --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
        --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
        --phrases /workspace/phrase-rl/data/screen_h$h.parquet \
        --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
        --repeats 3 \
        --out /workspace/phrase-rl/data/screen_out_h$h.parquet > /workspace/screen_h$h.log 2>&1 &
    done
    wait
    cd /workspace/phrase-rl
    .venv-gen/bin/python - <<PY || { echo "[val-loop] MERGE FAILED $ARM/$STAMP"; touch "results/val_screens/.done_${ARM}_${STAMP}"; continue; }
import pandas as pd
d = pd.concat([pd.read_parquet(f"data/screen_out_h{h}.parquet") for h in (1,2)], ignore_index=True)
d.to_parquet("$OUT", index=False)
s = d[d.arm=="tuned"].groupby("task").success.mean()*100
print(f"[val-loop] ARM $ARM $STAMP tuned per-task:", s.round(1).to_dict())
print(f"[val-loop] ARM $ARM $STAMP tuned pooled: {d[d.arm=='tuned'].success.mean()*100:.1f}% | base ride-along: {d[d.arm=='base'].success.mean()*100:.1f}%")
PY
    rm -f data/screen_out_h1.parquet data/screen_out_h2.parquet
    touch "results/val_screens/.done_${ARM}_${STAMP}"
    git add results/val_screens/ && git -c user.name=pod3 -c user.email=pod@runpod commit -q -m "val screen: arm $ARM $STAMP [pod]" && git -c user.name=pod3 -c user.email=pod@runpod pull -q --rebase --no-edit && git push -q
    did_work=1
  done
  [ $did_work = 0 ] && sleep 300
done
