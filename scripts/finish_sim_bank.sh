#!/usr/bin/env bash
# Resume the sim-bank chain from already-recorded trajectories: extract contexts,
# then score. Split out from run_sim_bank.sh so a failure in the cheap tail
# (extract, scoring) never costs the hours of recording that preceded it.
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
mark() { echo "[finish-sim $(date -u +%H:%M)] $*"; }

pick_python() {
  local want="$1"
  for v in /workspace/INT-ACT/.venv .venv-gen .venv; do
    [ -x "$v/bin/python" ] || continue
    if PYTHONPATH=/workspace/phrase-rl/src "$v/bin/python" -c "import $want" >/dev/null 2>&1; then
      echo "$v/bin/python"; return 0
    fi
  done
  return 1
}

TRAJ=data/sim_traj_val8
n=$(ls "$TRAJ" 2>/dev/null | wc -l | tr -d ' ')
[ "${n:-0}" -gt 0 ] || { mark "FATAL: no trajectories in $TRAJ"; exit 1; }
mark "$n recorded trajectory files"

PY=$(pick_python phrase_rl.sim_contexts_extract) || { mark "FATAL: no usable venv"; exit 1; }
mark "python: $PY"
PYTHONPATH=/workspace/phrase-rl/src $PY -m phrase_rl.sim_contexts_extract \
  --record-dir "$TRAJ" --out data/contexts_sim_val8.parquet --per-episode 4 \
  || { mark "FATAL extract"; exit 1; }

$PY - <<'PY'
import pandas as pd
d = pd.read_parquet("data/contexts_sim_val8.parquet")
k = "task" if "task" in d.columns else "instruction"
print("contexts per task:"); print(d.groupby(k).episode_index.nunique().to_string())
PY

TASK_KIND=sim SHARD=0 OF=1 IPC_DIR=/workspace/ipc_bank bash scripts/run_bank_scoring.sh
mark "sim half done; joining the training half as shard 1/2"

# e1 is working shard 0 of the training half; take shard 1 rather than idling
TASK_KIND=train SHARD=1 OF=2 IPC_DIR=/workspace/ipc_bank exec bash scripts/run_bank_scoring.sh
