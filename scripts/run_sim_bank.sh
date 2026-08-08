#!/usr/bin/env bash
# Pod-side: give the val8 simulator tasks (and their distractor scenes) the
# scoring contexts they lack, then score the sim half of the phrase bank.
#
#   bash scripts/run_sim_bank.sh
#
# Four native val8 tasks and the seven val-side distractor scenes have no
# context frames on any pod -- contexts_val_multit16 is Bridge real data, and the
# distractor scenes have never been rolled out. Both are produced the same way
# the OOV sim contexts were: record successful episodes, extract frames.
set -uo pipefail
export UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR="${VLA_DATA_DIR:-/workspace/vla_data}" VLA_LOG_DIR="${VLA_LOG_DIR:-/workspace/vla_log}" WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl
EPS="${EPS:-40}"
mark() { echo "[simbank $(date -u +%H:%M)] $*"; }

# never fight the leg that is already using the GPU
while pgrep -f "[p]hase0c_rollout" >/dev/null; do mark "waiting for the running leg"; sleep 300; done
for _ in $(seq 1 240); do pgrep -f "[g]it gc|[g]it repack" >/dev/null || break; sleep 30; done
timeout 600 git -c rebase.autoStash=true pull -q --rebase origin main 2>/dev/null \
  || { git rebase --abort 2>/dev/null; git fetch -q origin main && git reset --hard -q origin/main; }
mark "at $(git rev-parse --short HEAD)"

# the phrase spoken during recording is each env's own instruction, asked of the
# env rather than hardcoded, so a registry change cannot silently mis-caption
.venv-gen/bin/python - <<'PY'
import pandas as pd, simpler_env
TASKS = ["widowx_carrot_on_plate", "widowx_spoon_on_towel", "widowx_stack_cube",
         "widowx_put_eggplant_in_basket",
         "widowx_carrot_on_keyboard_distract", "widowx_carrot_on_plate_distract",
         "widowx_carrot_on_plate_lang_common_distract",
         "widowx_coke_can_on_plate_distract",
         "widowx_coke_can_on_plate_lang_common_distract",
         "widowx_spoon_on_towel_distract",
         "widowx_spoon_on_towel_lang_common_distract"]
rows = []
for t in TASKS:
    env = simpler_env.make(t)
    env.reset()
    ins = env.get_language_instruction()
    env.close()
    rows.append({"task": t, "phrase": ins, "arm": "original"})
    print(f"{t:48s} {ins!r}", flush=True)
pd.DataFrame(rows).to_parquet("data/phrases_sim_ctx.parquet", index=False)
PY
[ -s data/phrases_sim_ctx.parquet ] || { mark "FATAL: could not build phrases_sim_ctx"; exit 1; }

rm -f data/rollouts_simctx.parquet          # the rollout script APPENDS to --out
mark "recording $EPS episodes/task for context extraction"
cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/phrases_sim_ctx.parquet \
  --episode-ids $(seq 0 $((EPS - 1))) \
  --out /workspace/phrase-rl/data/rollouts_simctx.parquet \
  --record-dir /workspace/phrase-rl/data/sim_traj_val8 --record-success-only
rc=$?
cd /workspace/phrase-rl
mark "rollout rc=$rc"

.venv/bin/python -m phrase_rl.sim_contexts_extract --record-dir data/sim_traj_val8 \
  --out data/contexts_sim_val8.parquet --per-episode 4 || { mark "FATAL extract"; exit 1; }
.venv-gen/bin/python - <<'PY'
import pandas as pd
d = pd.read_parquet("data/contexts_sim_val8.parquet")
k = "task" if "task" in d.columns else "instruction"
print(d.groupby(k).episode_index.nunique().to_string())
PY

TASK_KIND=sim SHARD=0 OF=1 IPC_DIR=/workspace/ipc_bank exec bash scripts/run_bank_scoring.sh
