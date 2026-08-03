#!/usr/bin/env bash
# v9 — v8 fork + REPLAY + ADAPTIVE CxF (user 2026-08-02): ALL 2000 parents (no
# club filter); per-parent C = available same-instruction contexts (<=10),
# frames scaled to hold C*F ~= 40 (cap 16/ctx). 16 contexts/step, group-level
# replay buffer (16 replayed groups/update, window 50, clip 0.2, max reuse 6),
# warm-forked from v8@latest (~step 113). Reward UNCHANGED (grip-pure C=10 F=4 —
# user: the reward passed the CxF=40 discrimination exam; corpus is the open
# suspect, not the reward). Original v8 header follows:
#   * COLD START from frozen base Qwen (fresh zero-init LoRA; NO --init-adapter)
#   * NO TAGS: the arm-D gate externalizes regime detection, so the rewriter
#     never sees tier tags (v7a/v7f tag-era prompts are retired)
#   * tier mix nominal/benign/ert = 0.25/0.25/0.50 (v7a's proven mix)
#   * --input-dropout 0.5 (v6.1 slot dropout, TAG-FREE): half the prompts replace
#     the instruction slot with "infer the task from the context" — the model must
#     read the task from the trace (which quotes the source phrase). No tag involved;
#     the [withheld] tag was only an annotation when --tier-tags was on.
#   * grip-pure reward (blend-w 0.0), C=10 same-instruction club contexts, F=4
#   * beta = 0.15 (v7f's polish drift at beta=0.05 fired the Goodhart trigger:
#     41.7 -> 38.0 by step 80, IV-concentrated)
#   * checkpoints every 10 steps; pod eval pipeline runs stride 10, both
#     conditions, tag-free generation
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
timeout 90 git pull --no-edit || true

BETA="${BETA:-0.15}"

IPC_DIR=/workspace/ipc
mkdir -p "$IPC_DIR" results/checkpoints/phase2_v9
rm -f "$IPC_DIR"/*.req.json "$IPC_DIR"/*.done.json "$IPC_DIR"/*.failed.json \
      "$IPC_DIR"/*.err.txt "$IPC_DIR"/*.parquet "$IPC_DIR"/*.tmp

tmux kill-session -t score 2>/dev/null || true
tmux kill-session -t train 2>/dev/null || true

tmux new-session -d -s score \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME)\" ~/.bashrc || true)\"; export HF_HOME=\${HF_HOME:-/workspace/hf_cache}; cd /workspace/phrase-rl && \
   ${SCORE_PY:-.venv/bin/python} -m phrase_rl.phase2_score_server \
     --ipc-dir $IPC_DIR \
     --stats-contexts results/phrase_artifacts/chunk_stats.parquet \
     --verifier-ensemble results/checkpoints/verifier_reward_ensemble_4f.json \
   2>&1 | tee -a results/checkpoints/phase2_v9/score_server.log; \
   echo \"score server exited rc=\$?\"; sleep infinity'"

# --resume with NO --init-adapter: first launch finds no latest/ and cold-starts
# from base Qwen with a fresh LoRA (v7a's recipe); restarts resume v8's own latest.
tmux new-session -d -s train \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)\" ~/.bashrc || true)\"; export HF_HOME=\${HF_HOME:-/workspace/hf_cache}; cd /workspace/phrase-rl && \
   .venv-gen/bin/python -m phrase_rl.phase2_train \
     --ipc-dir $IPC_DIR --resume --init-adapter results/checkpoints/phase2_v8/latest \
     --ckpt-dir results/checkpoints/phase2_v9 \
     --reward-mode verifier --reward-frames ${REWARD_FRAMES:-4} \
     --source-mix 0.25,0.25,0.5 \
     --input-dropout 0.5 \
     --adaptive-contexts \
     --replay-groups 16 --replay-window 50 --replay-clip 0.2 --replay-max-reuse 6 \
     --reward-blend c4b --blend-w 0.0 --reward-contexts ${RCTX:-10} --club-contexts data/contexts_club.parquet \
     --gen-mode sample_single --update-rule grpo --no-gate \
     --gen-temp 1.0 --n-candidates 16 \
     --score-timeout 3600 \
     --beta $BETA --lr 7e-6 --kl-abort 1.2 \
     --contexts-per-step 16 --grad-accum-groups 2 \
     --val-every 21 --save-every 7 --probe-every 1000 --probe-samples 4 \
     --traces results/phrase_artifacts/cover35_teacher_train.parquet \
     --probe-contexts results/phrase_artifacts/contexts_probe8.parquet \
     --probe-traces results/phrase_artifacts/traces_probe8.parquet \
     --val-traces results/phrase_artifacts/rephrases_val_0b.parquet \
   2>&1 | tee -a results/checkpoints/phase2_v9/train.log; \
   echo \"trainer exited rc=\$?\"; sleep infinity'"

echo "V9 LAUNCHED (cold start, tag-free, input-dropout 0.5, replay, beta=$BETA, C=${RCTX:-10} F=${REWARD_FRAMES:-4})"
