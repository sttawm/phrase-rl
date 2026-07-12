#!/usr/bin/env bash
# Pod-3 CONTINUOUS red-team eval loop (user 2026-07-10: red-team at every val,
# x3 repeats — pod 3 is idle capital otherwise).
#
# Self-pacing: each round pulls the freshest best_val adapters; if the
# (flow_step, l2_step) pair is unchanged since the last round, sleep and re-check.
# Baselines (redteam_direct, base) don't depend on checkpoints: topped up to x3
# ONCE (marker file), then each round only rolls the two tuned arms:
#   2 arms x 4 tasks x 25 states x 3 reps = 600 episodes/round (~4-5h).
# Results: results/overnight/raw/rollouts_rt_f<flowstep>_l<l2step>.parquet, pushed.
set -uxo pipefail   # no -e: the loop must survive transient failures
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
P1_SCP="scp -q -r -o StrictHostKeyChecking=no -P 39484"
P2_SCP="scp -q -r -o StrictHostKeyChecking=no -P 22159 -i $HOME/.ssh/id_ed25519"
cd /workspace/phrase-rl
LAST_PAIR=""

step_of() { grep -oE '"step": ?[0-9]+' "$1/info.json" 2>/dev/null | grep -oE '[0-9]+' | head -1; }

while true; do
  git pull --no-edit || true

  # -- freshest adapters
  rm -rf /tmp/flow_new /tmp/l2_new
  $P1_SCP root@38.147.83.25:/workspace/phrase-rl/results/checkpoints/phase2/best_val /tmp/flow_new || { sleep 600; continue; }
  $P2_SCP root@69.30.85.249:/workspace/phrase-rl/results/checkpoints/phase2_l2/best_val /tmp/l2_new || { sleep 600; continue; }
  FSTEP=$(step_of /tmp/flow_new); LSTEP=$(step_of /tmp/l2_new)
  PAIR="f${FSTEP}_l${LSTEP}"
  if [ "$PAIR" = "$LAST_PAIR" ] || [ -f "results/overnight/raw/rollouts_rt_${PAIR}.parquet" ]; then
    echo "pair $PAIR already evaluated; sleeping"
    sleep 900
    continue
  fi
  echo "=== ROUND $PAIR ==="
  rm -rf results/checkpoints/eval_adapters/flow results/checkpoints/eval_adapters/l2
  mkdir -p results/checkpoints/eval_adapters
  mv /tmp/flow_new results/checkpoints/eval_adapters/flow
  mv /tmp/l2_new results/checkpoints/eval_adapters/l2

  # -- tuned phrases for this pair (Gemini traces cached in assets)
  cp -f results/phrase_artifacts/contexts_0c_tasks.parquet data/contexts_0c_tasks.parquet
  .venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
    --tasks results/phrase_artifacts/cover_gemini_tasks.parquet \
    --assets results/phrase_artifacts/redteam_eval_assets.parquet \
    --adapter results/checkpoints/eval_adapters/flow \
    --out data/phrases_rt_flow.parquet || { sleep 600; continue; }
  .venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
    --tasks results/phrase_artifacts/cover_gemini_tasks.parquet \
    --assets results/phrase_artifacts/redteam_eval_assets.parquet \
    --adapter results/checkpoints/eval_adapters/l2 \
    --out data/phrases_rt_l2.parquet || { sleep 600; continue; }

  .venv-gen/bin/python - <<PY || { sleep 600; continue; }
import os
import pandas as pd
fl = pd.read_parquet("data/phrases_rt_flow.parquet")
frames = [fl[fl.arm == "tuned"].assign(arm="tuned_flow")]
l2 = pd.read_parquet("data/phrases_rt_l2.parquet")
frames.append(l2[l2.arm == "tuned"].assign(arm="tuned_l2"))
if not os.path.exists("data/.baseline_x3_done"):
    # one-time: top the checkpoint-independent arms up to x3
    frames.append(fl[fl.arm == "original"].assign(arm="redteam_direct"))
    frames.append(fl[fl.arm == "base"])
out = pd.concat(frames, ignore_index=True).drop_duplicates(["task", "arm"])
out.to_parquet("data/phrases_rt_round.parquet", index=False)
print(out[["task", "arm", "phrase"]].to_string())
PY

  # -- rollouts x3 (baseline top-up seeds from the first round's x1 file if present)
  OUT="/workspace/phrase-rl/data/rollouts_rt_${PAIR}.parquet"
  if [ ! -f data/.baseline_x3_done ] && [ -f results/overnight/raw/rollouts_redteam_eval.parquet ]; then
    cp -n results/overnight/raw/rollouts_redteam_eval.parquet "$OUT" || true  # x1 baselines resume as rep 0
  fi
  cd /workspace/INT-ACT
  /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
    --int-act-root /workspace/INT-ACT \
    --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
    --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
    --phrases /workspace/phrase-rl/data/phrases_rt_round.parquet \
    --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 \
    --repeats 4 \
    --out "$OUT"
  RC=$?
  cd /workspace/phrase-rl
  [ $RC -ne 0 ] && { echo "rollout rc=$RC; retrying pair next cycle"; sleep 600; continue; }
  touch data/.baseline_x3_done

  # -- publish + table
  cp -f "$OUT" results/overnight/raw/
  cp -f data/phrases_rt_round.parquet "results/overnight/raw/phrases_rt_${PAIR}.parquet"
  git add results/overnight/raw/
  git -c user.name=pod3 -c user.email=pod@runpod commit -m "rt eval round ${PAIR}: tuned x3 [pod]" || true
  git -c user.name=pod3 -c user.email=pod@runpod pull --rebase --no-edit || true
  git push || true
  .venv-gen/bin/python - <<PY
import pandas as pd
d = pd.read_parquet("data/rollouts_rt_${PAIR}.parquet")
t = d.pivot_table(index="task", columns="arm", values="success", aggfunc="mean")
print((t * 100).round(1).to_string())
print("overall:"); print((d.groupby("arm")["success"].mean() * 100).round(1).to_string())
PY
  echo "ROUND-DONE $PAIR"
  LAST_PAIR="$PAIR"
done
