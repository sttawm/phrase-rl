#!/usr/bin/env bash
# v13 -- single-variable revision of v12 (user decision 2026-08-13):
#   * --source-mix 0,0.67,0.33 : ORIGINALS REMOVED from training inputs.
#     v12's copy rate tripled to 47% (identity wins on the canonical tier and
#     the bias bled across tiers; nat24 fell ~4pp while reward climbed).
#     Removing the tier where copying pays is the simplest intervention --
#     tried FIRST, alone, so the effect is attributable. No copy clamp, no
#     curriculum gate (both designed, held in reserve; --copy-guard-sim stays
#     available but off). copy_rate + n_copy_clamped telemetry stays -- the
#     center panel of the v12 chart is the metric this run must move.
# Everything else inherited from v12 (proxy_logit reward, C=10 collapsed,
# min-group-spread, sum ratio, gate ON, bplusimg prompt family, val-every 5).
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
mkdir -p "$IPC_DIR" results/checkpoints/phase2_v13
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
   2>&1 | tee -a results/checkpoints/phase2_v13/score_server.log; \
   echo \"score server exited rc=\$?\"; sleep infinity'"

tmux new-session -d -s train \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)\" ~/.bashrc || true)\"; export HF_HOME=\${HF_HOME:-/workspace/hf_cache}; cd /workspace/phrase-rl && \
   .venv-gen/bin/python -m phrase_rl.phase2_train \
     --ipc-dir $IPC_DIR --resume \
     --ckpt-dir results/checkpoints/phase2_v13 \
     --prompt-family bplusimg \
     --reward-mode verifier --reward-frames ${REWARD_FRAMES:-4} \
     --source-mix 0,0.67,0.33 \
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
     --val-every 5 --save-every 10 --probe-every 1000 --probe-samples 4 \
     --traces results/phrase_artifacts/naturals_v11_train.parquet \
     --probe-contexts results/phrase_artifacts/contexts_probe8.parquet \
     --probe-traces results/phrase_artifacts/traces_probe8.parquet \
     --val-traces results/phrase_artifacts/rephrases_val_0b.parquet \
   2>&1 | tee -a results/checkpoints/phase2_v13/train.log; \
   echo \"trainer exited rc=\$?\"; sleep infinity'"

echo "V13 LAUNCHED (proxy_logit, C=10 collapsed, min-spread 0.02, ratio=sum, gate ON, beta=$BETA)"
