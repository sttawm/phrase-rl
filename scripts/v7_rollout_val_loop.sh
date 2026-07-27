#!/usr/bin/env bash
# Async rollout-val for v7/v8. Polls the training pod's train_log.jsonl for new
# {"type":"probe"} rows and rolls, per checkpoint, TWO conditions:
#   greedy   : 4 tasks x 24 layouts x 2 reps of the greedy phrase           = 192 eps
#   sampled  : 4 tasks x 24 layouts x 1 rep each of 4 sampled phrases       = 384 eps (2x)
# (user 2026-07-26: sampled 2x, NOT the old 5x). Same 24 layouts, so greedy vs
# sampled is paired per cell; sampled = 4 independent policy draws per layout.
# RVAL_SAMPLES caps samples/probe (default 4). Appends pooled + sampled_pooled to
# results/analysis/v7_rollout_curve.jsonl. Zero cost to the training pod.
#   TRAIN_HOST_SSH="ssh -p PORT -i /root/.ssh/pod_relay root@HOST" bash v7_rollout_val_loop.sh
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl
CKPT_LOG=${CKPT_LOG:-/workspace/phrase-rl/results/checkpoints/phase2_v7/train_log.jsonl}
CURVE_FILE=${CURVE_FILE:-results/analysis/v7_rollout_curve.jsonl}
RUN_TAG=${RUN_TAG:-v7}
TRAIN_HOST_SSH=${TRAIN_HOST_SSH:?set TRAIN_HOST_SSH}
GEN=/workspace/phrase-rl/.venv-gen/bin/python
VLA=/workspace/INT-ACT/.venv/bin/python
ROLL=/workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py
CFG=config/experiment/simpler/pi0_finetune_bridge_ev.yaml
CKPT=juexzz/INTACT-pi0-finetune-rephrase-bridge
export CURVE_FILE RUN_TAG
mark() { echo "[rval $(date +%H:%M:%S)] $*" | tee -a /workspace/rval.log; }

# roll_one <phrases.parquet> <repeats> <out.parquet> : one phase0c call on layouts 0-23
roll_one() {
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases "$1" --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
    --repeats "$2" --out "$3" > "${3%.parquet}.log" 2>&1
  local rc=$?; cd /workspace/phrase-rl; return $rc
}

while true; do
  git pull -q --rebase 2>/dev/null
  $TRAIN_HOST_SSH "cat $CKPT_LOG" > /tmp/train_log.jsonl 2>/dev/null \
    || { mark "train host unreachable"; sleep 300; continue; }
  $GEN - <<'PYEOF'
import json, os, sys
import pandas as pd
done = set()
curve = os.environ["CURVE_FILE"]
if os.path.exists(curve):
    done = {json.loads(l)["step"] for l in open(curve)}
probes = [json.loads(l) for l in open("/tmp/train_log.jsonl") if '"probe"' in l]
todo = [p for p in probes if p["step"] not in done]
if not todo:
    sys.exit(3)
p = todo[-1]  # newest first; older backfill on later passes
greedy, sampled = [], []
for it in p["probes"]:
    task = it.get("task")
    g = it.get("greedy") or it.get("phrase")
    if task and g:
        greedy.append({"task": task, "arm": "v7_greedy", "phrase": g, "instruction": g})
    # up to RVAL_SAMPLES distinct samples -> 1 rep each on full grid (2x greedy)
    for sp in list(dict.fromkeys(it.get("samples") or []))[:int(os.environ.get("RVAL_SAMPLES", "4"))]:
        if task and sp:
            sampled.append({"task": task, "arm": "v7_sampled", "phrase": sp, "instruction": sp})
pd.DataFrame(greedy).to_parquet("data/rval_greedy.parquet", index=False)
pd.DataFrame(sampled).to_parquet("data/rval_sampled.parquet", index=False) if sampled \
    else pd.DataFrame(columns=["task","arm","phrase","instruction"]).to_parquet("data/rval_sampled.parquet", index=False)
json.dump({"step": p["step"], "n_greedy": len(greedy), "n_sampled": len(sampled)},
          open("/tmp/rval_meta.json", "w"))
PYEOF
  rc=$?
  [ $rc = 3 ] && { sleep 600; continue; }
  [ $rc != 0 ] && { mark "probe extraction failed rc=$rc"; sleep 600; continue; }
  STEP=$(python3 -c "import json;print(json.load(open('/tmp/rval_meta.json'))['step'])")
  NSAMP=$(python3 -c "import json;print(json.load(open('/tmp/rval_meta.json'))['n_sampled'])")
  mark "rolling step $STEP (greedy 2rep + sampled ${NSAMP}phr 1rep, equal 192-ep budget)"
  rm -f data/rval_greedy_out.parquet data/rval_sampled_out.parquet
  roll_one /workspace/phrase-rl/data/rval_greedy.parquet 2 /workspace/phrase-rl/data/rval_greedy_out.parquet &
  gp=$!
  sp=""
  if [ "$NSAMP" -gt 0 ]; then
    sleep 45  # stagger model loads
    roll_one /workspace/phrase-rl/data/rval_sampled.parquet 1 /workspace/phrase-rl/data/rval_sampled_out.parquet &
    sp=$!
  fi
  wfail=0; wait $gp || wfail=1; [ -n "$sp" ] && { wait $sp || wfail=1; }
  [ $wfail = 1 ] && { mark "ROLLOUT FAILED step $STEP"; sleep 300; continue; }
  STEP=$STEP $GEN - <<'PYEOF'
import glob, json, os
import pandas as pd
g = pd.read_parquet("data/rval_greedy_out.parquet")
rec = {"step": int(os.environ["STEP"]), "n": int(len(g)),
       "pooled": round(float(g.success.mean() * 100), 2),
       "per_task": {t: round(float(x.success.mean() * 100), 1) for t, x in g.groupby("task")}}
sf = "data/rval_sampled_out.parquet"
if os.path.exists(sf):
    s = pd.read_parquet(sf)
    if len(s):
        rec["sampled_n"] = int(len(s))
        rec["sampled_pooled"] = round(float(s.success.mean() * 100), 2)
        rec["sampled_per_task"] = {t: round(float(x.success.mean() * 100), 1) for t, x in s.groupby("task")}
with open(os.environ["CURVE_FILE"], "a") as f:
    f.write(json.dumps(rec) + "\n")
print("step", rec["step"], "greedy", rec["pooled"], "sampled", rec.get("sampled_pooled"))
PYEOF
  rc=$?
  [ $rc != 0 ] && { mark "MERGE FAILED step $STEP"; sleep 300; continue; }
  git add "$CURVE_FILE" && git commit -q -m "$RUN_TAG rollout-val step $STEP (greedy+sampled) [pod]" \
    && git pull -q --rebase && git push -q
  mark "DONE step $STEP"
done
