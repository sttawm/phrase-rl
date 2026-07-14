#!/usr/bin/env bash
# Rollout-reward RL arm (pod 2 handover after the L2 plateau, user standing order).
# Reward = ACTUAL SIMPLER success rate; contexts = the 96 (task, layout) cells of the
# val grid; NO val pass (user: train on val states, report training success; final
# eval ONCE on held-out tasks). Same trainer otherwise: 32 candidates, z-advantages,
# top-8 positives, grad-accum 4, KL leash.
#
# Step budget: 2 ctx x ~10 scored candidates x 2 reps ~= 40 episodes ~= 5-6 min/step.
# 96 contexts / cps 2 = 48 steps/epoch (~4.5h/epoch).
set -euxo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/uv_cache}" UV_LINK_MODE=copy
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl
git pull --no-edit

IPC_DIR=/workspace/ipc_rollout
CKPT_DIR=results/checkpoints/phase2_rollout
mkdir -p "$IPC_DIR" "$CKPT_DIR" data
rm -f "$IPC_DIR"/*.req.json "$IPC_DIR"/*.done.json "$IPC_DIR"/*.failed.json "$IPC_DIR"/*.parquet "$IPC_DIR"/*.err.txt 2>/dev/null || true

# 1) sim training contexts + per-task traces (one-time)
if [ ! -f data/contexts_sim_rollout_train.parquet ]; then
  /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/sim_train_contexts.py \
    --int-act-root /workspace/INT-ACT --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 \
    --out /workspace/phrase-rl/data/contexts_sim_rollout_train.parquet
fi
if [ ! -f data/traces_sim_rollout.parquet ]; then
  .venv-gen/bin/python - <<'PY'
import pandas as pd
ctx = pd.read_parquet("data/contexts_sim_rollout_train.parquet")
tr = pd.read_parquet("results/phrase_artifacts/traces_0c_tasks.parquet")
g = pd.read_parquet("results/phrase_artifacts/cover_gemini_tasks.parquet")
ep2task = {}
for r in g.itertuples():
    ep2task[int(r.episode_index)] = None  # placeholder; task mapping via contexts file
tasks_ctx = pd.read_parquet("results/phrase_artifacts/contexts_0c_tasks.parquet")
epi2task = {int(r.episode_index): r.task for r in tasks_ctx.itertuples()}
task2trace = {}
for r in tr.itertuples():
    t = epi2task.get(int(r.episode_index))
    if t: task2trace[t] = str(r.trace)
rows = [{"episode_index": int(r.episode_index), "t": 0, "trace": task2trace[r.task]}
        for r in ctx.itertuples()]
pd.DataFrame(rows).to_parquet("data/traces_sim_rollout.parquet", index=False)
print(f"{len(rows)} trace rows over {len(task2trace)} tasks")
PY
fi

tmux kill-session -t rollout_server 2>/dev/null || true
tmux kill-session -t train 2>/dev/null || true

tmux new-session -d -s rollout_server \
  "bash -lc 'export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline HF_HOME=/workspace/hf_cache; \
   /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase2_rollout_server.py \
     --int-act-root /workspace/INT-ACT --ipc-dir $IPC_DIR \
   2>&1 | tee -a $PWD/$CKPT_DIR/rollout_server.log; echo \"server exited rc=\$?\"; sleep infinity'"

tmux new-session -d -s train \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY|UV_CACHE_DIR|UV_LINK_MODE)\" ~/.bashrc || true)\"; cd /workspace/phrase-rl && \
   CUDA_VISIBLE_DEVICES=0 .venv-gen/bin/python -m phrase_rl.phase2_train \
     --ipc-dir $IPC_DIR --ckpt-dir $CKPT_DIR --resume \
     --reward-mode rollout --rollout-reps 2 --update-rule grpo \
     --train-contexts data/contexts_sim_rollout_train.parquet \
     --traces data/traces_sim_rollout.parquet \
     --probe-contexts results/phrase_artifacts/contexts_0c_tasks.parquet \
     --probe-traces results/phrase_artifacts/traces_0c_tasks.parquet --probe-every 10 \
     --beta 0.15 --lr 7e-6 --kl-abort 1.2 \
     --contexts-per-step 2 --n-candidates 16 --grad-accum-groups 4 --max-pos-per-ctx 8 \
     --no-gate --val-every 0 --save-every 10 \
     --score-timeout 1800 \
   2>&1 | tee -a $CKPT_DIR/train.log; \
   echo \"trainer exited rc=\$? (0=clean 4=score timeout 5=server error 6=KL abort)\"; sleep infinity'"

echo "rollout-RL launched:"
echo "  server : tmux attach -t rollout_server (INT-ACT venv, $IPC_DIR)"
echo "  trainer: tmux attach -t train           (log: $CKPT_DIR/train_log.jsonl)"
echo "  NOTE: --val-every 0 by design — reward IS deployment success; watch cand_loss_mean"
echo "  (= -success rate) fall in the step records. Final eval: held-out tasks, once."
