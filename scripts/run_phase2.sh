#!/usr/bin/env bash
# Phase 2 pod orchestration: frozen-pi0 CRN score server (.venv, INTACT-era
# lerobot) + advantage-weighted Qwen trainer (.venv-gen, transformers>=5).
# The two venvs cannot be merged (dependency conflict), so they talk through
# file-based IPC in $IPC_DIR (protocol documented in phase2_train.py).
# Both run in tmux and survive SSH drops. Idempotent: re-running restarts both
# sessions and the trainer resumes from results/checkpoints/phase2/latest.
set -euxo pipefail
export UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl

if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "GEMINI_API_KEY not set (the faithfulness gate needs it) — add it to ~/.bashrc"
  exit 1
fi

IPC_DIR=/workspace/ipc
mkdir -p "$IPC_DIR" results/checkpoints/phase2
# stale IPC files from a crashed run would be mistaken for live requests/responses
rm -f "$IPC_DIR"/*.req.json "$IPC_DIR"/*.done.json "$IPC_DIR"/*.failed.json \
      "$IPC_DIR"/*.err.txt "$IPC_DIR"/*.parquet "$IPC_DIR"/*.tmp

# GPU split per EXPERIMENT.md "Compute": GPU 0 = reward server, GPU 1 = phrase
# model; share GPU 0 on a single-GPU pod (pi0 ~7GB + Qwen ~18GB fits on 48GB).
NGPU=$(nvidia-smi -L | wc -l)
SCORE_GPU=0
TRAIN_GPU=$(( NGPU >= 2 ? 1 : 0 ))

tmux kill-session -t score 2>/dev/null || true
tmux kill-session -t train 2>/dev/null || true

# NEVER `uv run` here — it re-syncs and uninstalls packages. Direct venv pythons.
tmux new-session -d -s score \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY|UV_CACHE_DIR|UV_LINK_MODE)\" ~/.bashrc || true)\"; cd /workspace/phrase-rl && \
   CUDA_VISIBLE_DEVICES=$SCORE_GPU .venv/bin/python -m phrase_rl.phase2_score_server \
     --ipc-dir $IPC_DIR \
   2>&1 | tee -a results/checkpoints/phase2/score_server.log; \
   echo \"score server exited rc=\$?\"; sleep infinity'"

tmux new-session -d -s train \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY|UV_CACHE_DIR|UV_LINK_MODE)\" ~/.bashrc || true)\"; cd /workspace/phrase-rl && \
   CUDA_VISIBLE_DEVICES=$TRAIN_GPU .venv-gen/bin/python -m phrase_rl.phase2_train \
     --ipc-dir $IPC_DIR --resume \
     --beta 0.15 --lr 7e-6 --kl-abort 1.2 --contexts-per-step 6 --no-gate \
     --n-candidates 32 --grad-accum-groups 4 \
     --val-every 40 --save-every 10 --probe-every 10 \
     ${JUDGE_BACKEND:+--judge-backend $JUDGE_BACKEND --judge-model gemini-3.5-flash} \
     ${INIT_ADAPTER:+--init-adapter $INIT_ADAPTER} \
     --traces results/phrase_artifacts/cover35_teacher_train.parquet \
     --probe-contexts results/phrase_artifacts/contexts_0c_tasks.parquet \
     --probe-traces results/phrase_artifacts/traces_0c_tasks.parquet --probe-every 25 \
     --val-traces results/phrase_artifacts/rephrases_val_0b.parquet  # INTERIM: old-format val traces (cover35 val blocked on credits, ~\$3); keep constant for the whole run — do NOT swap mid-run \
   2>&1 | tee -a results/checkpoints/phase2/train.log; \
   echo \"trainer exited rc=\$? (0=clean/interrupted 3=gate unavailable 4=score timeout 5=score server error)\"; \
   sleep infinity'"

set +x
cat <<EOF

Phase 2 launched (score on GPU $SCORE_GPU, train on GPU $TRAIN_GPU).
  score server : tmux attach -t score    log: results/checkpoints/phase2/score_server.log
  trainer      : tmux attach -t train    log: results/checkpoints/phase2/train.log
  step metrics : tail -f results/checkpoints/phase2/train_log.jsonl
  val/best     : results/checkpoints/metrics.json  (key "phase2")
  checkpoints  : results/checkpoints/phase2/{latest,best_val,final}

Trainer exit codes: 0 = clean or SIGTERM/Ctrl-C (checkpointed, resumable),
                    3 = faithfulness gate unavailable (fix judge API, re-run),
                    4 = score-server timeout (check 'score' session, re-run),
                    5 = score-server job error (see .err.txt in train log, re-run).
Re-run this script to restart both; the trainer resumes from latest.
EOF
