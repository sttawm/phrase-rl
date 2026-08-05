#!/usr/bin/env bash
# Generic single-leg roller for the parallel eval fleet (2026-08-05).
# Env: POD (label), ARM, PHRASES (repo-relative parquet), MODE=sealed|rr.
#   sealed: episode-ids 0-23 x 12 reps, n=3456 -> sealed_ladder_cells.jsonl
#   rr:     episode-ids 0-11 x 1 rep,   n=2304 -> rephrase_robustness.jsonl
# Waits (git pull loop) for PHRASES if absent. Runs a 1-episode SMOKE roll
# first (renderer + policy sanity) and aborts loudly on renderer fallback.
set -uo pipefail
export HF_HOME=/workspace/hf_cache VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
mark() { echo "[$POD $(date -u +%H:%M)] $*" | tee -a /workspace/leg.log; }
OUT_SL=results/analysis/sealed_ladder_cells.jsonl
OUT_RR=results/analysis/rephrase_robustness.jsonl

CKPT="${CKPT:-juexzz/INTACT-pi0-finetune-rephrase-bridge}"
LAYSET="${LAYSET:-first}"
if [ "$MODE" = sealed ]; then OUT=$OUT_SL; EPS="$(seq 0 23)"; REPS=12; N=3456
elif [ "$LAYSET" = second ]; then OUT=$OUT_RR; EPS="$(seq 12 23)"; REPS=1; N=2304
else OUT=$OUT_RR; EPS="$(seq 0 11)"; REPS=1; N=2304; fi
grep -q "\"arm\": \"$ARM\"" $OUT 2>/dev/null && { mark "skip $ARM (merged)"; exit 0; }

tries=0
until [ -f "$PHRASES" ]; do
  tries=$((tries+1)); [ $tries -gt 60 ] && { mark "WAIT-TIMEOUT $ARM"; exit 1; }
  mark "waiting for $PHRASES ($tries)"
  timeout 200 git -c rebase.autoStash=true pull -q --rebase 2>/dev/null; sleep 300
done

mark "SMOKE roll for $ARM"
rm -f data/smoke_out.parquet data/smoke_phrases.parquet
/workspace/INT-ACT/.venv/bin/python -c "
import pandas as pd
pd.read_parquet('$PHRASES').head(1).to_parquet('data/smoke_phrases.parquet', index=False)"
cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt "$CKPT" --phrases /workspace/phrase-rl/data/smoke_phrases.parquet \
  --episode-ids 0 --repeats 1 \
  --out /workspace/phrase-rl/data/smoke_out.parquet > /workspace/smoke.log 2>&1 \
  || { grep -iqE "ExtensionNotPresent|llvmpipe|vulkan" /workspace/smoke.log && mark "SMOKE FAIL RENDERER $ARM" || mark "SMOKE FAIL $ARM"; exit 1; }
cd /workspace/phrase-rl
mark "SMOKE-OK; roll $ARM (n=$N)"
rm -f data/leg_out.parquet
cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt "$CKPT" --phrases /workspace/phrase-rl/$PHRASES \
  --episode-ids $EPS --repeats $REPS \
  --out /workspace/phrase-rl/data/leg_out.parquet > /workspace/leg_roll.log 2>&1 \
  || { mark "ROLL FAIL $ARM"; exit 1; }
cd /workspace/phrase-rl
# post-roll recheck: another pod may have merged this arm while we rolled
timeout 200 git -c rebase.autoStash=true pull -q --rebase 2>/dev/null
grep -q "\"arm\": \"$ARM\"" $OUT 2>/dev/null && { mark "skip-merge $ARM (landed elsewhere mid-roll)"; exit 0; }
ARMN=$ARM MODEN=$MODE NN=$N LAYSETN=$LAYSET /workspace/INT-ACT/.venv/bin/python - <<'PYEOF' || { mark "MERGE FAIL $ARM"; exit 1; }
import json
import os

import pandas as pd

g = pd.read_parquet("data/leg_out.parquet")
n = int(os.environ["NN"])
assert len(g) == n, f"n={len(g)} != {n}"
rec = {"arm": os.environ["ARMN"], "n": n,
       "pooled": round(float(g.success.mean() * 100), 2),
       "per_task": {t: round(float(x.success.mean() * 100), 1) for t, x in g.groupby("task")}}
if os.environ["MODEN"] == "rr":
    lays = sorted(int(e) for e in g.episode_id.unique())
    exp = list(range(12, 24)) if os.environ.get("LAYSETN") == "second" else list(range(12))
    assert lays == exp, f"layout set {lays} != {exp}"
    rec["layouts"] = lays
    rec["reps"] = 1
    per = g.groupby(["task", "phrase"]).success.mean().mul(100).round(2)
    rec["per_rephrase"] = {f"{t}|{p}": float(v) for (t, p), v in per.items()}
    out = "results/analysis/rephrase_robustness.jsonl"
else:
    out = "results/analysis/sealed_ladder_cells.jsonl"
open(out, "a").write(json.dumps(rec) + "\n")
print(rec["arm"], rec["pooled"])
PYEOF
suffix=$([ "$MODE" = sealed ] && echo x12 || echo x1)
cp data/leg_out.parquet results/sealed/${ARM}_${suffix}.parquet
ok=0
for i in 1 2 3 4 5; do
  timeout 300 bash -c "git add $OUT results/sealed/${ARM}_${suffix}.parquet && git commit -q -m 'leg: $ARM [$POD]' && git -c rebase.autoStash=true pull -q --rebase && git push -q" && { ok=1; break; }
  mark "push attempt $i failed"; sleep 120
done
[ $ok = 1 ] && mark "DONE $ARM" || mark "DONE-UNPUSHED $ARM"
