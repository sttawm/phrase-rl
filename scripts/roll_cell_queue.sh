#!/usr/bin/env bash
# Generic sealed-cell queue roller. Args: POD_TAG then triples "parquet:arm".
# Waits (git pull loop) for each phrases parquet, rolls 12x24x12, merges to
# results/analysis/sealed_ladder_cells.jsonl, x12 copy, push-retries.
set -uo pipefail
export HF_HOME=/workspace/hf_cache VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
POD=$1; shift
OUT=results/analysis/sealed_ladder_cells.jsonl
mark() { echo "[cellq-$POD $(date -u +%H:%M)] $*" | tee -a /workspace/cellq.log; }
for spec in "$@"; do
  ph="${spec%%:*}"; arm="${spec##*:}"
  grep -q "\"arm\": \"$arm\"" $OUT 2>/dev/null && { mark "skip $arm"; continue; }
  tries=0
  until [ -f "$ph" ]; do
    tries=$((tries+1)); [ $tries -gt 48 ] && { mark "WAIT-TIMEOUT $arm"; continue 2; }
    timeout 200 git -c rebase.autoStash=true pull -q --rebase 2>/dev/null; sleep 300
  done
  mark "roll $arm (12x24x12)"
  rm -f data/cellq_out.parquet
  cd /workspace/INT-ACT
  /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py --int-act-root /workspace/INT-ACT --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --phrases /workspace/phrase-rl/$ph --episode-ids $(seq 0 23) --repeats 12 --out /workspace/phrase-rl/data/cellq_out.parquet > /workspace/cellq_$arm.log 2>&1 || { mark "ROLL FAIL $arm"; cd /workspace/phrase-rl; continue; }
  cd /workspace/phrase-rl
  ARMN=$arm /workspace/INT-ACT/.venv/bin/python - <<'PYEOF' || { mark "MERGE FAIL $arm"; continue; }
import json, os
import pandas as pd
g = pd.read_parquet("data/cellq_out.parquet")
assert len(g) == 3456, f"n={len(g)}"
rec = {"arm": os.environ["ARMN"], "n": 3456, "pooled": round(float(g.success.mean()*100), 2),
       "per_task": {t: round(float(x.success.mean()*100), 1) for t, x in g.groupby("task")}}
open("results/analysis/sealed_ladder_cells.jsonl", "a").write(json.dumps(rec) + "\n")
print(rec["arm"], rec["pooled"])
PYEOF
  cp data/cellq_out.parquet results/sealed/${arm}_x12.parquet
  ok=0
  for i in 1 2 3 4 5; do
    timeout 300 bash -c "git add $OUT results/sealed/${arm}_x12.parquet && git commit -q -m 'cell: $arm x12 [$POD]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" && { ok=1; break; }
    mark "push attempt $i failed for $arm"; sleep 120
  done
  [ $ok = 1 ] && mark "DONE $arm" || mark "DONE-UNPUSHED $arm"
done
mark "ALL-CELLS-DONE-$POD"
