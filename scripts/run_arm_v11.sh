#!/usr/bin/env bash
# v11 — RL on PROMPT B (user design 2026-08-04/05): CoVer skeleton with ALL
# rules-like content + few-shots removed, NO image (Gemini trace is the only
# scene channel), single-line output contract. Purpose: causal test of the
# anchoring hypothesis (v8/v9 froze under the few-shot-anchored scaffold) +
# family-fair comparison to the prompting ladder.
#   * COLD START from frozen base Qwen (fresh zero-init LoRA; NO --init-adapter)
#   * --prompt-family bplus (build_single_phrase_prefix_bare everywhere:
#     generation, update prefix, probes, val)
#   * scoring HALVED per the fine-pair exam (user 2026-08-05): C=5 F=4 = 20
#     evals/parent (exam: 69-70 pairwise vs 73 at C=10; frames matter more
#     than contexts at half budget, so F stays 4). Grip-pure kept.
#   * v9c step economics: 8 fresh + 8 replay, accum 1 (one 256-cand update per
#     step); text-only prompts -> projected ~10-11 min/step
#   * tag-free, input-dropout 0.5, beta 0.15, lr 7e-6, KL-abort 1.2
#   * save-every 10, val-every 20 (user 2026-08-05: ~10-step evals with slack at the ~11min step); per-step latest/ + grads/rng persistence
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
timeout 90 git pull --no-edit || true

BETA="${BETA:-0.15}"

IPC_DIR=/workspace/ipc
mkdir -p "$IPC_DIR" results/checkpoints/phase2_v11
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
   2>&1 | tee -a results/checkpoints/phase2_v11/score_server.log; \
   echo \"score server exited rc=\$?\"; sleep infinity'"

tmux new-session -d -s train \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)\" ~/.bashrc || true)\"; export HF_HOME=\${HF_HOME:-/workspace/hf_cache}; cd /workspace/phrase-rl && \
   .venv-gen/bin/python -m phrase_rl.phase2_train \
     --ipc-dir $IPC_DIR --resume \
     --ckpt-dir results/checkpoints/phase2_v11 \
     --prompt-family bplus \
     --reward-mode verifier --reward-frames ${REWARD_FRAMES:-4} \
     --source-mix 0.25,0.5,0.25 \
     --input-dropout 0.5 \
     --adaptive-contexts \
     --replay-groups 8 --replay-window 50 --replay-clip 0.2 --replay-max-reuse 6 \
     --reward-blend c4b --blend-w 0.25 --reward-contexts ${RCTX:-5} --club-contexts data/contexts_club.parquet \
     --gen-mode sample_single --update-rule grpo --no-gate \
     --gen-temp 1.0 --n-candidates 16 \
     --score-timeout 3600 \
     --beta $BETA --lr 7e-6 --kl-abort 1.2 \
     --contexts-per-step 8 --grad-accum-groups 1 \
     --val-every 20 --save-every 10 --probe-every 1000 --probe-samples 4 \
     --traces results/phrase_artifacts/naturals_v11_train.parquet \
     --probe-contexts results/phrase_artifacts/contexts_probe8.parquet \
     --probe-traces results/phrase_artifacts/traces_probe8.parquet \
     --val-traces results/phrase_artifacts/rephrases_val_0b.parquet \
   2>&1 | tee -a results/checkpoints/phase2_v11/train.log; \
   echo \"trainer exited rc=\$?\"; sleep infinity'"

echo "V10 LAUNCHED (prompt B, cold start, C=5 F=4, 8+8, accum 1, beta=$BETA)"
