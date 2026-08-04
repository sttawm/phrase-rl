#!/usr/bin/env bash
# Amendment 20 roll: rephrase16_rulesv4_gemini on the standard executor,
# layouts 0-11 x 1 rep (matches the robustness protocol). Gated on the rrq
# session (arm 2 pi0base) finishing.
set -uo pipefail
export HF_HOME=/workspace/hf_cache VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
mark() { echo "[rrules-pod5 $(date -u +%H:%M)] $*" | tee -a /workspace/rrq.log; }
PH=results/sealed/ph_sealed_rephrase16_rulesv4.parquet
OUT=results/analysis/rephrase_robustness.jsonl

while tmux has-session -t rrq 2>/dev/null; do sleep 300; done
mark "rrq finished — rolling rules-v4 rephrase arm"
grep -q '"arm": "rephrase16_rulesv4_gemini"' $OUT 2>/dev/null && { mark "already merged; exit"; exit 0; }
[ -f "$PH" ] || { mark "ABORT: $PH missing"; exit 1; }
rm -f data/rr_out.parquet
cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --phrases /workspace/phrase-rl/$PH \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 --repeats 1 \
  --out /workspace/phrase-rl/data/rr_out.parquet > /workspace/rr_rulesv4.log 2>&1 \
  || { mark "ROLL FAIL rulesv4"; exit 1; }
cd /workspace/phrase-rl
/workspace/INT-ACT/.venv/bin/python - <<'PYEOF' || { mark "MERGE FAIL rulesv4"; exit 1; }
import json

import pandas as pd

g = pd.read_parquet("data/rr_out.parquet")
assert len(g) == 2304, f"n={len(g)}"
assert sorted(g.episode_id.unique().tolist()) == list(range(12))
per = g.groupby(["task", "phrase"]).success.mean().mul(100).round(2)
rec = {"arm": "rephrase16_rulesv4_gemini", "n": 2304, "layouts": list(range(12)), "reps": 1,
       "pooled": round(float(g.success.mean() * 100), 2),
       "per_task": {t: round(float(x.success.mean() * 100), 1) for t, x in g.groupby("task")},
       "per_rephrase": {f"{t}|{p}": float(v) for (t, p), v in per.items()}}
open("results/analysis/rephrase_robustness.jsonl", "a").write(json.dumps(rec) + "\n")
print(rec["arm"], rec["pooled"])
PYEOF
cp data/rr_out.parquet results/sealed/rephrase16_rulesv4_x1.parquet
ok=0
for i in 1 2 3 4 5; do
  timeout 300 bash -c "git add $OUT results/sealed/rephrase16_rulesv4_x1.parquet && git commit -q -m 'rephrase-robustness: rulesv4_gemini layouts 0-11 [pod5]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" && { ok=1; break; }
  mark "push attempt $i failed"; sleep 120
done
[ $ok = 1 ] && mark "DONE rulesv4" || mark "DONE-UNPUSHED rulesv4"
mark "RRULES-COMPLETE"
