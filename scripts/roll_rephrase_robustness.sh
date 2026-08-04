#!/usr/bin/env bash
# Rephrase-robustness overnight queue (PREREG Amendment 19), pod5.
# Gate: waits for the v3nom cellq session to finish, then rolls BOTH pi0
# executors on the K=16 rephrase set. LAYOUTS 0-11 ONLY (recorded in every
# output row + the merge record), 1 rep => 12 tasks x 16 x 12 = 2,304 eps/arm
# (~5.7h each). Arms: rephrase16_pi0rephrase (our executor) then
# rephrase16_pi0base (non-rephrase-augmented INTACT pi0 — its first appearance).
set -uo pipefail
export HF_HOME=/workspace/hf_cache VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
mark() { echo "[rr-pod5 $(date -u +%H:%M)] $*" | tee -a /workspace/rrq.log; }
PH=results/sealed/ph_sealed_rephrase16.parquet
OUT=results/analysis/rephrase_robustness.jsonl

while tmux has-session -t cellq 2>/dev/null; do sleep 300; done
mark "cellq finished — starting rephrase-robustness queue"
timeout 120 git -c rebase.autoStash=true pull -q --rebase 2>/dev/null
[ -f "$PH" ] || { mark "ABORT: $PH missing"; exit 1; }

for spec in "juexzz/INTACT-pi0-finetune-rephrase-bridge:pi0rephrase" "juexzz/INTACT-pi0-finetune-bridge:pi0base"; do
  ck="${spec%%:*}"; tag="${spec##*:}"
  grep -q "\"arm\": \"rephrase16_$tag\"" $OUT 2>/dev/null && { mark "skip $tag"; continue; }
  mark "roll $tag (12 tasks x 16 rephrases x layouts 0-11 x 1 rep)"
  rm -f data/rr_out.parquet
  cd /workspace/INT-ACT
  /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
    --int-act-root /workspace/INT-ACT --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
    --ckpt "$ck" --phrases /workspace/phrase-rl/$PH \
    --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 --repeats 1 \
    --out /workspace/phrase-rl/data/rr_out.parquet > /workspace/rr_$tag.log 2>&1 \
    || { mark "ROLL FAIL $tag"; cd /workspace/phrase-rl; continue; }
  cd /workspace/phrase-rl
  TAG=$tag /workspace/INT-ACT/.venv/bin/python - <<'PYEOF' || { mark "MERGE FAIL $tag"; continue; }
import json
import os

import pandas as pd

g = pd.read_parquet("data/rr_out.parquet")
assert len(g) == 2304, f"n={len(g)}"
assert sorted(g.episode_id.unique().tolist()) == list(range(12)), "layout set != 0-11"
tag = os.environ["TAG"]
per = g.groupby(["task", "phrase"]).success.mean().mul(100).round(2)
rec = {"arm": f"rephrase16_{tag}", "n": 2304, "layouts": list(range(12)), "reps": 1,
       "pooled": round(float(g.success.mean() * 100), 2),
       "per_task": {t: round(float(x.success.mean() * 100), 1) for t, x in g.groupby("task")},
       "per_rephrase": {f"{t}|{p}": float(v) for (t, p), v in per.items()}}
open("results/analysis/rephrase_robustness.jsonl", "a").write(json.dumps(rec) + "\n")
print(rec["arm"], rec["pooled"])
PYEOF
  cp data/rr_out.parquet results/sealed/rephrase16_${tag}_x1.parquet
  ok=0
  for i in 1 2 3 4 5; do
    timeout 300 bash -c "git add $OUT results/sealed/rephrase16_${tag}_x1.parquet && git commit -q -m 'rephrase-robustness: $tag layouts 0-11 [pod5]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" && { ok=1; break; }
    mark "push attempt $i failed for $tag"; sleep 120
  done
  [ $ok = 1 ] && mark "DONE $tag" || mark "DONE-UNPUSHED $tag"
done
mark "RR-QUEUE-COMPLETE"
