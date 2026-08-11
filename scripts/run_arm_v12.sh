#!/usr/bin/env bash
# v12 — magnitude-reward, multiplicity-collapsed GRPO (ALGORITHM-RL-V12.md).
# Deltas from v11, each tied to a v11 measurement:
#   * --reward-blend proxy_logit : fixed-coefficient calibrated logit
#     0.4445*z + 11.3193*(-grip), clipped; rank01 manufactured gradient from
#     float noise (29.4% of duplicate sets carried spurious rank spread)
#   * --reward-contexts 10       : paid for by --collapse-duplicate-scoring
#     (score distinct strings once, re-expand before the transform; exact)
#   * --min-group-spread 0.02    : GRPO normalizes noise-only groups to unit
#     advantage variance; skip them
#   * --ratio-mode sum           : replay ratio was true_ratio^(1/n); clip
#     never bound
#   * gate ON (no --no-gate)     : needs GEMINI_API_KEY in ~/.bashrc
# Unchanged: sample_single (on-policy, no importance weights), K=16 (collapse
# falsified as the flat-val cause), signed full-group advantages, beta 0.15,
# lr 7e-6, kl-abort 1.2, F=4, save-every 10, source-mix 0.25/0.5/0.25.
# COLD START from frozen base Qwen (fresh zero-init LoRA), prompt B+.
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
timeout 90 git pull --no-edit || true

BETA="${BETA:-0.15}"

IPC_DIR="${IPC_DIR:-/workspace/ipc}"
case "$IPC_DIR" in
  ""|"/"|"/workspace"|"/root"|"/tmp"|*..*) echo "unsafe IPC_DIR=$IPC_DIR" >&2; exit 1;;
esac
[ -d "$IPC_DIR" ] || mkdir -p "$IPC_DIR"
mkdir -p "$IPC_DIR" results/checkpoints/phase2_v12
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
   2>&1 | tee -a results/checkpoints/phase2_v12/score_server.log; \
   echo \"score server exited rc=\$?\"; sleep infinity'"

tmux new-session -d -s train \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)\" ~/.bashrc || true)\"; export HF_HOME=\${HF_HOME:-/workspace/hf_cache}; cd /workspace/phrase-rl && \
   .venv-gen/bin/python -m phrase_rl.phase2_train \
     --ipc-dir $IPC_DIR --resume \
     --ckpt-dir results/checkpoints/phase2_v12 \
     --prompt-family bplus \
     --reward-mode verifier --reward-frames ${REWARD_FRAMES:-4} \
     --source-mix 0.25,0.5,0.25 \
     --input-dropout 0.5 \
     --adaptive-contexts \
     --replay-groups 8 --replay-window 50 --replay-clip 0.2 --replay-max-reuse 6 \
     --reward-blend proxy_logit --reward-contexts ${RCTX:-10} --club-contexts data/contexts_club.parquet \
     --collapse-duplicate-scoring --min-group-spread 0.02 --ratio-mode sum \
     --gen-mode sample_single --update-rule grpo \
     --gen-temp 1.0 --n-candidates 16 \
     --score-timeout 3600 \
     --beta $BETA --lr 7e-6 --kl-abort 1.2 \
     --contexts-per-step 8 --grad-accum-groups 1 \
     --val-every 20 --save-every 10 --probe-every 1000 --probe-samples 4 \
     --traces results/phrase_artifacts/naturals_v11_train.parquet \
     --probe-contexts results/phrase_artifacts/contexts_probe8.parquet \
     --probe-traces results/phrase_artifacts/traces_probe8.parquet \
     --val-traces results/phrase_artifacts/rephrases_val_0b.parquet \
   2>&1 | tee -a results/checkpoints/phase2_v12/train.log; \
   echo \"trainer exited rc=\$?\"; sleep infinity'"

echo "V12 LAUNCHED (proxy_logit, C=10 collapsed, min-spread 0.02, ratio=sum, gate ON, beta=$BETA)"
