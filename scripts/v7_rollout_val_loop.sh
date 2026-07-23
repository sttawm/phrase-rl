#!/usr/bin/env bash
# Async rollout-val for v7/v8 (user 2026-07-24: "make the eval ~300-500 eps").
# Polls the training pod's train_log.jsonl for new {"type":"probe"} rows,
# rolls each probe's greedy deployment phrases on the val-8 contexts at
# 24 layouts x2 (n=384/checkpoint), appends to results/analysis/v7_rollout_curve.jsonl.
# Zero cost to the training pod. Needs a Vulkan-capable pod (POD-SETUP.md).
#   TRAIN_HOST_SSH="ssh -p 47005 -i /root/.ssh/pod_relay root@103.196.86.39" \
#   bash scripts/v7_rollout_val_loop.sh
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
CKPT_LOG=/workspace/phrase-rl/results/checkpoints/phase2_v7/train_log.jsonl
TRAIN_HOST_SSH=${TRAIN_HOST_SSH:?set TRAIN_HOST_SSH}
mark() { echo "[rval $(date +%H:%M:%S)] $*" | tee -a /workspace/rval.log; }

while true; do
  git pull -q --rebase 2>/dev/null
  $TRAIN_HOST_SSH "cat $CKPT_LOG" > /tmp/train_log.jsonl 2>/dev/null || { mark "train host unreachable"; sleep 300; continue; }
  .venv-gen/bin/python - <<'PYEOF'
import json, os, subprocess, sys
import pandas as pd
done = set()
curve = "results/analysis/v7_rollout_curve.jsonl"
if os.path.exists(curve):
    done = {json.loads(l)["step"] for l in open(curve)}
probes = [json.loads(l) for l in open("/tmp/train_log.jsonl") if '"probe"' in l]
todo = [p for p in probes if p["step"] not in done]
if not todo:
    sys.exit(3)
p = todo[-1]  # newest first; older ones fill in later passes
rows = []
for it in p["probes"]:
    task, phrase = it.get("task"), it.get("greedy") or it.get("phrase")
    if task and phrase:
        rows.append({"task": task, "arm": f"v7_step{p['step']}", "phrase": phrase, "instruction": phrase})
pd.DataFrame(rows).to_parquet("data/rval_phrases.parquet", index=False)
json.dump({"step": p["step"], "n_phrases": len(rows)}, open("/tmp/rval_meta.json","w"))
PYEOF
  rc=$?
  [ $rc = 3 ] && { sleep 600; continue; }
  [ $rc != 0 ] && { mark "probe extraction failed rc=$rc"; sleep 600; continue; }
  STEP=$(python3 -c "import json;print(json.load(open('/tmp/rval_meta.json'))['step'])")
  mark "rolling probe step $STEP"
  rm -f data/rval_out_w*.parquet
  cd /workspace/INT-ACT
  pids=""
  for w in 0 1 2; do
    sleep 45
    /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
      --int-act-root /workspace/INT-ACT \
      --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
      --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
      --phrases /workspace/phrase-rl/data/rval_phrases.parquet \
      --worker-index $w --num-workers 3 \
      --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
      --repeats 2 \
      --out /workspace/phrase-rl/data/rval_out_w$w.parquet > /workspace/rval_w$w.log 2>&1 &
    pids="$pids $!"
  done
  wfail=0; for pid in $pids; do wait $pid || wfail=1; done
  cd /workspace/phrase-rl
  [ $wfail = 1 ] && { mark "ROLLOUT FAILED step $STEP"; sleep 300; continue; }
  STEP=$STEP .venv-gen/bin/python - <<'PYEOF'
import glob, json, os
import pandas as pd
d = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob("data/rval_out_w*.parquet"))], ignore_index=True)
rec = {"step": int(os.environ["STEP"]), "n": int(len(d)),
       "pooled": round(float(d.success.mean()*100), 2),
       "per_task": {t: round(float(g.success.mean()*100), 1) for t, g in d.groupby("task")}}
with open("results/analysis/v7_rollout_curve.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("step", rec["step"], "pooled", rec["pooled"], "n", rec["n"])
PYEOF
  git add results/analysis/v7_rollout_curve.jsonl && git commit -q -m "v7 rollout-val step $STEP [pod]" && git pull -q --rebase && git push -q
  mark "DONE step $STEP"
done
