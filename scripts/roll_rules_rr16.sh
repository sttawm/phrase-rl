#!/usr/bin/env bash
# Amendment 23 queue (pod5), gated on the prompt-B queue (pbq session) finishing:
#   0. Qwen rules-v4 generation over rephrase16 (GPU)
#   1. roll rr16_rulesv4_qwen   (192 x layouts 0-11 x 1 = 2,304)
#   2. roll rr16_rulesv4_claude (2,304)
set -uo pipefail
export HF_HOME=/workspace/hf_cache VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
mark() { echo "[a23-pod5 $(date -u +%H:%M)] $*" | tee -a /workspace/pbq.log; }
RR=results/analysis/rephrase_robustness.jsonl

while tmux has-session -t pbq 2>/dev/null; do sleep 300; done
mark "pbq finished — Amendment-23 queue starting"

if [ ! -f results/sealed/ph_rr16_rulesv4_qwen.parquet ]; then
  mark "gen qwen rules-v4 rr16"
  FINAL_EVAL=1 .venv-gen/bin/python scripts/gen_rules_qwen_rr16.py > /workspace/a23_qwengen.log 2>&1 \
    || mark "QWEN RULES GEN FAIL"
  timeout 300 bash -c "git add results/sealed/ph_rr16_rulesv4_qwen.parquet && git commit -q -m 'a23: qwen rules-v4 rr16 phrases' && git -c rebase.autoStash=true pull -q --rebase && git push -q" || mark "PUSH-DEFERRED qwen rules phrases"
fi

roll_rr() { # $1 phrases parquet, $2 arm
  grep -q "\"arm\": \"$2\"" $RR 2>/dev/null && { mark "skip $2"; return 0; }
  [ -f "$1" ] || { mark "MISSING $1 — skip $2"; return 1; }
  mark "roll $2 (192 x layouts 0-11 x 1)"
  rm -f data/rr_out.parquet
  cd /workspace/INT-ACT
  /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
    --int-act-root /workspace/INT-ACT --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
    --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --phrases /workspace/phrase-rl/$1 \
    --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 --repeats 1 \
    --out /workspace/phrase-rl/data/rr_out.parquet > /workspace/a23_$2.log 2>&1 \
    || { mark "ROLL FAIL $2"; cd /workspace/phrase-rl; return 1; }
  cd /workspace/phrase-rl
  ARMN=$2 /workspace/INT-ACT/.venv/bin/python - <<'PYEOF' || { mark "MERGE FAIL $ARMN"; return 1; }
import json
import os

import pandas as pd

g = pd.read_parquet("data/rr_out.parquet")
assert len(g) == 2304, f"n={len(g)}"
assert sorted(g.episode_id.unique().tolist()) == list(range(12))
per = g.groupby(["task", "phrase"]).success.mean().mul(100).round(2)
rec = {"arm": os.environ["ARMN"], "n": 2304, "layouts": list(range(12)), "reps": 1,
       "pooled": round(float(g.success.mean() * 100), 2),
       "per_task": {t: round(float(x.success.mean() * 100), 1) for t, x in g.groupby("task")},
       "per_rephrase": {f"{t}|{p}": float(v) for (t, p), v in per.items()}}
open("results/analysis/rephrase_robustness.jsonl", "a").write(json.dumps(rec) + "\n")
print(rec["arm"], rec["pooled"])
PYEOF
  cp data/rr_out.parquet results/sealed/$2_x1.parquet
  for i in 1 2 3 4 5; do
    timeout 300 bash -c "git add $RR results/sealed/$2_x1.parquet && git commit -q -m 'rr arm: $2 [pod5]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" && { mark "DONE $2"; return 0; }
    sleep 120
  done
  mark "DONE-UNPUSHED $2"
}

roll_rr results/sealed/ph_rr16_rulesv4_qwen.parquet rr16_rulesv4_qwen
roll_rr results/sealed/ph_rr16_rulesv4_claude.parquet rr16_rulesv4_claude
mark "A23-COMPLETE"
