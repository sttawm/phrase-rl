#!/usr/bin/env bash
# v7c — LR-sweep fork (lr 7e-6 -> 2e-5, beta inherits v7b 0.05) of v7 (user 2026-07-26). Hypothesis: v7's KL collapsed to
# ~0.04 (abort 1.2, beta 0.15) as rollout plateaued — over-regularized. v7b warm-
# starts the POLICY from v7's rollout peak (step 140, greedy 58.3) via --init-adapter
# (fresh optimizer/step), keeps the KL reference = frozen base (unchanged), and drops
# beta 0.15 -> 0.05 to let the policy concentrate mass on its good samples. Everything
# else identical to v7 (C4b reward, F=16, 25/25/50 mix, 0.5 dropout, grpo, lr 7e-6,
# probe-samples 4). JUDGE ON REAL ROLLOUTS (proxy can improve via reward-hacking).
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
timeout 90 git pull --no-edit || true

FORK_CKPT="${FORK_CKPT:-results/checkpoints/phase2_v7/step_0140}"
BETA="${BETA:-0.05}"
LR="${LR:-2e-5}"
[ -f "$FORK_CKPT/adapter_config.json" ] || { echo "fork ckpt missing: $FORK_CKPT"; exit 1; }

IPC_DIR="${IPC_DIR:-/workspace/ipc}"
# the rm below expands $IPC_DIR into a glob: refuse anything that would make it
# scan a root or shared directory, even though every caller passes a literal
case "$IPC_DIR" in
  ""|"/"|"/workspace"|"/root"|"/tmp"|*..*) echo "unsafe IPC_DIR=$IPC_DIR" >&2; exit 1;;
esac
[ -d "$IPC_DIR" ] || mkdir -p "$IPC_DIR"
mkdir -p "$IPC_DIR" results/checkpoints/phase2_v7c
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
   2>&1 | tee -a results/checkpoints/phase2_v7c/score_server.log; \
   echo \"score server exited rc=\$?\"; sleep infinity'"

# --resume first (so restarts of v7b resume its OWN latest); only --init-adapter on a
# cold start. phase2_train prefers latest/ when both are present.
tmux new-session -d -s train \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)\" ~/.bashrc || true)\"; export HF_HOME=\${HF_HOME:-/workspace/hf_cache}; cd /workspace/phrase-rl && \
   .venv-gen/bin/python -m phrase_rl.phase2_train \
     --ipc-dir $IPC_DIR --resume --init-adapter $FORK_CKPT \
     --ckpt-dir results/checkpoints/phase2_v7c \
     --reward-mode verifier --reward-frames ${REWARD_FRAMES:-16} \
     --source-mix 0.25,0.25,0.5 \
     --input-dropout 0.5 \
     --tier-tags \
     --reward-blend c4b --blend-w 0.25 \
     --gen-mode sample_single --update-rule grpo --no-gate \
     --gen-temp 1.0 --n-candidates 16 \
     --score-timeout 1800 \
     --beta $BETA --lr $LR --kl-abort 1.2 \
     --contexts-per-step 8 --grad-accum-groups 6 \
     --val-every 20 --save-every 10 --probe-every 25 --probe-samples 4 \
     --traces results/phrase_artifacts/cover35_teacher_train.parquet \
     --probe-contexts results/phrase_artifacts/contexts_0c_tasks.parquet \
     --probe-traces results/phrase_artifacts/traces_0c_tasks.parquet \
     --val-traces results/phrase_artifacts/rephrases_val_0b.parquet \
   2>&1 | tee -a results/checkpoints/phase2_v7c/train.log; \
   echo \"trainer exited rc=\$?\"; sleep infinity'"

echo "V7b LAUNCHED (fork=$FORK_CKPT, beta=$BETA, gen_mode=sample_single)"
