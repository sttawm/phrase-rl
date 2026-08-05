#!/usr/bin/env bash
# v10 step-0 anchor queue (Amendment 21), pod5. Gated on the rules-rephrase
# roll (rrules session) finishing. Generates prompt-B anchors (frozen base
# Qwen, both conditions), then rolls each on the FULL sealed protocol
# (12x24x12 = 3,456 eps/leg) -> sealed_ladder_cells.jsonl as
# v10step0_polish / v10step0_repair. These are v10's pre-registered step-0
# AND the prompt-B bare-baseline row.
set -uo pipefail
export HF_HOME=/workspace/hf_cache VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
mark() { echo "[v10anchor-pod5 $(date -u +%H:%M)] $*" | tee -a /workspace/v10anchor.log; }
OUT=results/analysis/sealed_ladder_cells.jsonl

while tmux has-session -t rrules 2>/dev/null; do sleep 300; done
mark "rrules finished — generating prompt-B anchors"
FINAL_EVAL=1 .venv-gen/bin/python scripts/gen_v10_anchors.py > /workspace/v10anchor_gen.log 2>&1 \
  || { mark "GEN FAIL"; exit 1; }
grep -c PREFLIGHT /workspace/v10anchor_gen.log | xargs -I{} echo "[v10anchor-pod5] {} phrases preflighted" | tee -a /workspace/v10anchor.log
timeout 300 bash -c "git add results/sealed/ph_sealed_v10step0_*.parquet && git commit -q -m 'v10 step-0 anchor phrases (prompt B, frozen base Qwen)' && git -c rebase.autoStash=true pull -q --rebase && git push -q" || mark "PUSH-DEFERRED phrases"

for cond in polish repair; do
  arm="v10step0_$cond"
  grep -q "\"arm\": \"$arm\"" $OUT 2>/dev/null && { mark "skip $arm"; continue; }
  mark "roll $arm (12x24x12)"
  rm -f data/cellq_out.parquet
  cd /workspace/INT-ACT
  /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
    --int-act-root /workspace/INT-ACT --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
    --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
    --phrases /workspace/phrase-rl/results/sealed/ph_sealed_${arm}.parquet \
    --episode-ids $(seq 0 23) --repeats 12 \
    --out /workspace/phrase-rl/data/cellq_out.parquet > /workspace/v10anchor_$cond.log 2>&1 \
    || { mark "ROLL FAIL $arm"; cd /workspace/phrase-rl; continue; }
  cd /workspace/phrase-rl
  ARMN=$arm /workspace/INT-ACT/.venv/bin/python - <<'PYEOF' || { mark "MERGE FAIL $arm"; continue; }
import json
import os

import pandas as pd

g = pd.read_parquet("data/cellq_out.parquet")
assert len(g) == 3456, f"n={len(g)}"
rec = {"arm": os.environ["ARMN"], "n": 3456, "pooled": round(float(g.success.mean() * 100), 2),
       "per_task": {t: round(float(x.success.mean() * 100), 1) for t, x in g.groupby("task")}}
open("results/analysis/sealed_ladder_cells.jsonl", "a").write(json.dumps(rec) + "\n")
print(rec["arm"], rec["pooled"])
PYEOF
  cp data/cellq_out.parquet results/sealed/${arm}_x12.parquet
  ok=0
  for i in 1 2 3 4 5; do
    timeout 300 bash -c "git add $OUT results/sealed/${arm}_x12.parquet && git commit -q -m 'cell: $arm x12 [pod5]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" && { ok=1; break; }
    mark "push attempt $i failed for $arm"; sleep 120
  done
  [ $ok = 1 ] && mark "DONE $arm" || mark "DONE-UNPUSHED $arm"
done
mark "V10-ANCHORS-COMPLETE"
