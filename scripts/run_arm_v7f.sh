#!/usr/bin/env bash
# v7f — v7e with C=10 reward contexts (fork of v7e @ step_0020, single-variable change).
# Rationale (2026-07-28, fine+coarse grids): C=10 is the knee of the measured C-axis —
# 8->10 is the steepest segment (+4.6pp grip sign-acc, coarse F=4), 10->16 buys only
# +2.1pp while costing ~60% more scoring time (step ~37min -> ~26min, +48% steps/day).
# F·C = 4x10 = 40 sits at the top of the theoretically-optimal 12-32 band.
# Reward unchanged: grip-pure (blend-w 0.0) averaged over C same-instruction club
# episodes x F=4 frames; parents = traced club episodes; ensemble-z logged as Goodhart
# tripwire (pre-registered signature: grip up while ens-z down). beta=0.05, lr 7e-6.
# Provenance note: steps 1-20 of the policy were trained under C=16 (v7e); C is a
# variance knob, not a bias knob, so the prefix is benign — ledgered in EXPERIMENT.md.
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
timeout 90 git pull --no-edit || true

FORK_CKPT="${FORK_CKPT:-results/checkpoints/phase2_v7e/step_0020}"
BETA="${BETA:-0.05}"
[ -f "$FORK_CKPT/adapter_config.json" ] || { echo "fork ckpt missing: $FORK_CKPT"; exit 1; }

IPC_DIR=/workspace/ipc
mkdir -p "$IPC_DIR" results/checkpoints/phase2_v7f
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
   2>&1 | tee -a results/checkpoints/phase2_v7f/score_server.log; \
   echo \"score server exited rc=\$?\"; sleep infinity'"

# --resume first (so restarts of v7f resume its OWN latest); only --init-adapter on a
# cold start. phase2_train prefers latest/ when both are present.
tmux new-session -d -s train \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)\" ~/.bashrc || true)\"; export HF_HOME=\${HF_HOME:-/workspace/hf_cache}; cd /workspace/phrase-rl && \
   .venv-gen/bin/python -m phrase_rl.phase2_train \
     --ipc-dir $IPC_DIR --resume --init-adapter $FORK_CKPT \
     --ckpt-dir results/checkpoints/phase2_v7f \
     --reward-mode verifier --reward-frames ${REWARD_FRAMES:-4} \
     --source-mix 0.25,0.25,0.5 \
     --input-dropout 0.5 \
     --tier-tags \
     --reward-blend c4b --blend-w 0.0 --reward-contexts ${RCTX:-10} --club-contexts data/contexts_club.parquet \
     --gen-mode sample_single --update-rule grpo --no-gate \
     --gen-temp 1.0 --n-candidates 16 \
     --score-timeout 3600 \
     --beta $BETA --lr 7e-6 --kl-abort 1.2 \
     --contexts-per-step 8 --grad-accum-groups 6 \
     --val-every 20 --save-every 10 --probe-every 25 --probe-samples 4 \
     --traces results/phrase_artifacts/cover35_teacher_train.parquet \
     --probe-contexts results/phrase_artifacts/contexts_probe8.parquet \
     --probe-traces results/phrase_artifacts/traces_probe8.parquet \
     --val-traces results/phrase_artifacts/rephrases_val_0b.parquet \
   2>&1 | tee -a results/checkpoints/phase2_v7f/train.log; \
   echo \"trainer exited rc=\$?\"; sleep infinity'"

echo "V7F LAUNCHED (fork=$FORK_CKPT, beta=$BETA, reward-contexts=${RCTX:-10})"
