#!/usr/bin/env bash
# v5 A/B launcher — VERIFIER REWARD (locked 2026-07-16: 5-seed calibrated
# both_nodiff ensemble). One script, arm selected by $1:
#   run_arm.sh A   -> Arm A: 16-list generation (structural diversity), SIGNED
#                     advantages (grpo update rule), gateless
#   run_arm.sh B   -> Arm B: true p_single sampling (on-policy GRPO), no dedupe,
#                     signed advantages, gateless
# Both: reward = verifier ensemble, nominal inputs + source-aug (trainer default),
# cached traces, resumable. Re-run to restart; trainer resumes from latest.
set -euxo pipefail
ARM="${1:?usage: run_arm.sh A|B}"
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
git pull --no-edit || true

case "$ARM" in
  A) GEN_MODE=list ;;
  B) GEN_MODE=sample_single ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac

IPC_DIR=/workspace/ipc
mkdir -p "$IPC_DIR" results/checkpoints/phase2
rm -f "$IPC_DIR"/*.req.json "$IPC_DIR"/*.done.json "$IPC_DIR"/*.failed.json \
      "$IPC_DIR"/*.err.txt "$IPC_DIR"/*.parquet "$IPC_DIR"/*.tmp

tmux kill-session -t score 2>/dev/null || true
tmux kill-session -t train 2>/dev/null || true

tmux new-session -d -s score \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME)\" ~/.bashrc || true)\"; export HF_HOME=\${HF_HOME:-/workspace/hf_cache}; cd /workspace/phrase-rl && \
   .venv/bin/python -m phrase_rl.phase2_score_server \
     --ipc-dir $IPC_DIR \
     --stats-contexts results/phrase_artifacts/chunk_stats.parquet \
     --verifier-ensemble results/checkpoints/verifier_reward_ensemble.json \
   2>&1 | tee -a results/checkpoints/phase2/score_server.log; \
   echo \"score server exited rc=\$?\"; sleep infinity'"

tmux new-session -d -s train \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)\" ~/.bashrc || true)\"; export HF_HOME=\${HF_HOME:-/workspace/hf_cache}; cd /workspace/phrase-rl && \
   .venv-gen/bin/python -m phrase_rl.phase2_train \
     --ipc-dir $IPC_DIR --resume \
     --reward-mode verifier \
     --gen-mode $GEN_MODE --update-rule grpo --no-gate \
     --gen-temp 1.0 --n-candidates 16 \
     --beta 0.15 --lr 7e-6 --kl-abort 1.2 \
     --contexts-per-step 6 --grad-accum-groups 4 \
     --val-every 40 --save-every 10 --probe-every 25 \
     --traces results/phrase_artifacts/cover35_teacher_train.parquet \
     --probe-contexts results/phrase_artifacts/contexts_0c_tasks.parquet \
     --probe-traces results/phrase_artifacts/traces_0c_tasks.parquet \
     --val-traces results/phrase_artifacts/rephrases_val_0b.parquet \
   2>&1 | tee -a results/checkpoints/phase2/train.log; \
   echo \"trainer exited rc=\$?\"; sleep infinity'"

echo "ARM $ARM LAUNCHED (gen_mode=$GEN_MODE, reward=verifier ensemble, signed advantages)"
