#!/usr/bin/env bash
# Pod-2 overnight: grow the rollout-success label set for the reward bake-off.
# Gemini-free. Stages: [setup] -> base 0c-redo grid (tmux 'grid', resumes from
# partial) + concurrently score/decode the redo phrases -> deep-seed selection
# (extremes) -> 15 extra seeds on ~40 phrases -> publish. All stages resumable.
set -euxo pipefail
export UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR="${VLA_DATA_DIR:-/workspace/vla_data}" VLA_LOG_DIR="${VLA_LOG_DIR:-/workspace/vla_log}" WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl
git pull --no-edit || true

[ -d .venv ] || bash scripts/setup_pod.sh
[ -d /workspace/INT-ACT ] || bash scripts/setup_simpler.sh
cp -n results/overnight/raw/phrases_0c_redo.parquet data/ 2>/dev/null || true

# ---- base grid (long pole) in its own session ----
tmux kill-session -t grid 2>/dev/null || true
tmux new-session -d -s grid "bash -c '
  export HF_HOME=/workspace/hf_cache VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
  eval \"\$(grep -E \"^export HF_TOKEN\" ~/.bashrc)\"
  cd /workspace/INT-ACT && /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
    --int-act-root /workspace/INT-ACT \
    --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
    --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
    --phrases /workspace/phrase-rl/data/phrases_0c_redo.parquet \
    --episode-ids 0 1 2 3 4 5 6 7 8 9 \
    --out /workspace/phrase-rl/data/rollouts_0c_redo.parquet > /workspace/grid.log 2>&1'"

# ---- concurrent: flow-score + decode the redo phrases (feature arms) ----
source .venv/bin/activate
python -m phrase_rl.phase0c_score --contexts data/contexts_0c_match.parquet \
  --phrases data/phrases_0c_redo.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --out data/scores_0c_redo.parquet
python -m phrase_rl.pi0_decode --contexts data/contexts_0c_match.parquet \
  --phrases data/phrases_0c_redo.parquet --stats-contexts data/contexts_train.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --out data/decoded_0c_redo.parquet
deactivate
mkdir -p results/overnight/raw && cp -f data/scores_0c_redo.parquet data/decoded_0c_redo.parquet results/overnight/raw/
git add results/overnight && git -c user.name=pod2 -c user.email=pod@runpod commit -m "pod2: redo flow scores + decoded actions [pod]" || true
git -c user.name=pod2 -c user.email=pod@runpod pull --rebase --no-edit || true; git push || true

# ---- wait for base grid, then deep-seed ----
while tmux has-session -t grid 2>/dev/null; do sleep 120; done
cp -f data/rollouts_0c_redo.parquet results/overnight/raw/ 2>/dev/null || true

.venv/bin/python - <<'PY'
import pandas as pd, numpy as np
sc = pd.read_parquet("data/scores_0c_redo.parquet")
ro = pd.read_parquet("data/rollouts_0c_redo.parquet")
rows = []
for task in ro.task.unique():
    tsucc = ro[ro.task == task].groupby("phrase")["success"].mean()
    tl = sc[sc.task == task].groupby("phrase")["loss"].mean() if (sc.task == task).any() else None
    if tl is not None and len(tl) > 10:
        order = tl.sort_values()  # flow extremes + middles
        pick = list(order.index[:4]) + list(order.index[-4:]) + list(order.index[len(order)//2 - 1: len(order)//2 + 1])
    else:  # eggplant: success extremes from the base grid
        order = tsucc.sort_values()
        pick = list(order.index[:4]) + list(order.index[-4:]) + list(order.index[len(order)//2 - 1: len(order)//2 + 1])
    arm = pd.read_parquet("data/phrases_0c_redo.parquet")
    amap = dict(zip(arm[arm.task == task].phrase, arm[arm.task == task].arm))
    rows += [{"task": task, "arm": amap.get(p, "?"), "phrase": p} for p in dict.fromkeys(pick)]
df = pd.DataFrame(rows)
df.to_parquet("data/phrases_deepseed.parquet", index=False)
print(df.groupby("task").size().to_string())
PY

cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/phrases_deepseed.parquet \
  --episode-ids 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
  --out /workspace/phrase-rl/data/rollouts_deepseed.parquet
cd /workspace/phrase-rl

cp -f data/rollouts_0c_redo.parquet data/rollouts_deepseed.parquet data/phrases_deepseed.parquet results/overnight/raw/
git add results/overnight && git -c user.name=pod2 -c user.email=pod@runpod commit -m "pod2: base grid + deep-seed rollout labels [pod]" || true
git -c user.name=pod2 -c user.email=pod@runpod pull --rebase --no-edit || true; git push || true
echo "POD2 OVERNIGHT DONE"
