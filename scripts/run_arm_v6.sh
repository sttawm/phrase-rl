#!/usr/bin/env bash
# v6 A/B launcher — NATIVE-4F REWARD + TIERED INPUTS (EXPERIMENT.md "v6 run spec").
#   run_arm_v6.sh A   -> 16-list gen (SOURCE-conditioned: conditioning consistency),
#                        signed advantages, gateless
#   run_arm_v6.sh B   -> true p_single sampling (on-policy GRPO), no dedupe
# Both: reward = native-4f calibrated 5-seed ensemble; inputs 25% nominal / 25%
# benign / 50% hostile-ERT with tier-matched traces; FROM SCRATCH in a fresh
# ckpt dir (v5's results/checkpoints/phase2 is never touched).
set -euxo pipefail
ARM="${1:?usage: run_arm_v6.sh A|B}"
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
mkdir -p "$IPC_DIR" results/checkpoints/phase2_v6
rm -f "$IPC_DIR"/*.req.json "$IPC_DIR"/*.done.json "$IPC_DIR"/*.failed.json \
      "$IPC_DIR"/*.err.txt "$IPC_DIR"/*.parquet "$IPC_DIR"/*.tmp

tmux kill-session -t score 2>/dev/null || true
tmux kill-session -t train 2>/dev/null || true

tmux new-session -d -s score \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME)\" ~/.bashrc || true)\"; export HF_HOME=\${HF_HOME:-/workspace/hf_cache}; cd /workspace/phrase-rl && \
   .venv/bin/python -m phrase_rl.phase2_score_server \
     --ipc-dir $IPC_DIR \
     --stats-contexts results/phrase_artifacts/chunk_stats.parquet \
     --verifier-ensemble results/checkpoints/verifier_reward_ensemble_4f.json \
   2>&1 | tee -a results/checkpoints/phase2_v6/score_server.log; \
   echo \"score server exited rc=\$?\"; sleep infinity'"

tmux new-session -d -s train \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)\" ~/.bashrc || true)\"; export HF_HOME=\${HF_HOME:-/workspace/hf_cache}; cd /workspace/phrase-rl && \
   .venv-gen/bin/python -m phrase_rl.phase2_train \
     --ipc-dir $IPC_DIR --resume \
     --ckpt-dir results/checkpoints/phase2_v6 \
     --reward-mode verifier --reward-frames ${REWARD_FRAMES:-4} \
     --source-mix 0.25,0.25,0.5 \
     --input-dropout 0.3333 \
     --tier-tags \
     --gen-mode $GEN_MODE --update-rule grpo --no-gate \
     --gen-temp 1.0 --n-candidates 16 \
     --beta 0.15 --lr 7e-6 --kl-abort 1.2 \
     --contexts-per-step 8 --grad-accum-groups 6 \
     --val-every 20 --save-every 10 --probe-every 25 \
     --traces results/phrase_artifacts/cover35_teacher_train.parquet \
     --probe-contexts results/phrase_artifacts/contexts_0c_tasks.parquet \
     --probe-traces results/phrase_artifacts/traces_0c_tasks.parquet \
     --val-traces results/phrase_artifacts/rephrases_val_0b.parquet \
   2>&1 | tee -a results/checkpoints/phase2_v6/train.log; \
   echo \"trainer exited rc=\$?\"; sleep infinity'"

echo "V6 ARM $ARM LAUNCHED (native-4f reward, source mix 25/25/50, gen_mode=$GEN_MODE)"
