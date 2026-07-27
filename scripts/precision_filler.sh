#!/usr/bin/env bash
# Idle-time precision filler for pod5 (user 2026-07-27: pod5 ~25% utilized).
# Re-rolls PIVOTAL checkpoints with extra reps to shrink gate-deciding SEs
# (n=192 -> combined n>=576 halves SE ~3.6pp -> ~2.1pp). Yields to live rval:
# never starts a job while phase0c is running; checks between jobs.
# Queue: v7b probe points (phrases from the fetched v7b train log) + v7a pivots
# (phrases regenerated from local adapters). Appends results/analysis/
# v7_precision.jsonl records {run, step, condition, n, successes}.
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
GEN=/workspace/phrase-rl/.venv-gen/bin/python
VLA=/workspace/INT-ACT/.venv/bin/python
ROLL=/workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py
CFG=config/experiment/simpler/pi0_finetune_bridge_ev.yaml
CKPT=juexzz/INTACT-pi0-finetune-rephrase-bridge
OUT=results/analysis/v7_precision.jsonl
mark() { echo "[prec $(date -u +%H:%M)] $*" | tee -a /workspace/precision.log; }

busy() { pgrep -f "[p]hase0c_rollout" >/dev/null; }

wait_idle() {
  while busy; do sleep 300; done
  # small grace so rval can claim the GPU first if a probe just arrived
  sleep 120
  ! busy
}

roll() {  # <phrases.parquet> <reps> <out.parquet>
  cd /workspace/INT-ACT
  $VLA $ROLL --int-act-root /workspace/INT-ACT --config $CFG --ckpt $CKPT \
    --phrases "$1" --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
    --repeats "$2" --out "$3" > "${3%.parquet}.log" 2>&1
  local rc=$?; cd /workspace/phrase-rl; return $rc
}

record() {  # <run> <step> <condition> <parquet>
  RUN=$1 STEP=$2 COND=$3 PQ=$4 $GEN - <<'PYEOF'
import json, os
import pandas as pd
d = pd.read_parquet(os.environ["PQ"])
rec = {"run": os.environ["RUN"], "step": int(os.environ["STEP"]),
       "condition": os.environ["COND"], "n": int(len(d)),
       "successes": int(d.success.sum()),
       "per_task": {t: [int(x.success.sum()), int(len(x))] for t, x in d.groupby("task")}}
with open("results/analysis/v7_precision.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print(rec["run"], rec["step"], rec["condition"], f"{100*rec['successes']/rec['n']:.1f}%")
PYEOF
}

done_already() { grep -q "\"run\": \"$1\", \"step\": $2, \"condition\": \"$3\"" $OUT 2>/dev/null; }

# --- job implementations ---
job_v7b_probe() {  # <step> — phrases from the fetched v7b train log (rval keeps /tmp/train_log.jsonl fresh)
  local s=$1
  STEP=$s $GEN - <<'PYEOF' || return 1
import json, os
import pandas as pd
step = int(os.environ["STEP"])
probes = [json.loads(l) for l in open("/tmp/train_log.jsonl") if '"probe"' in l]
p = next((x for x in probes if x["step"] == step), None)
assert p, f"no probe at {step}"
g = [{"task": it["task"], "arm": "g", "phrase": it.get("greedy") or it["phrase"], "instruction": it.get("greedy") or it["phrase"]} for it in p["probes"]]
s_ = [{"task": it["task"], "arm": "s", "phrase": sp, "instruction": sp}
      for it in p["probes"] for sp in list(dict.fromkeys(it.get("samples") or []))[:2]]
pd.DataFrame(g).to_parquet("data/prec_g.parquet", index=False)
pd.DataFrame(s_).to_parquet("data/prec_s.parquet", index=False)
PYEOF
  wait_idle || return 1
  done_already v7b $s greedy || { roll data/prec_g.parquet 4 data/prec_g_out.parquet && record v7b $s greedy data/prec_g_out.parquet; }
  wait_idle || return 1
  done_already v7b $s sampled || { roll data/prec_s.parquet 2 data/prec_s_out.parquet && record v7b $s sampled data/prec_s_out.parquet; }
}

job_adapter() {  # <run> <step> <adapter_dir> — regen phrases from weights, roll both
  local run=$1 s=$2 d=$3
  [ -f "$d/adapter_model.safetensors" ] || { mark "no adapter $d"; return 1; }
  wait_idle || return 1
  $GEN scripts/gen_ckpt_phrases.py "$d" 2 > /workspace/prec_gen_${run}_$s.log 2>&1 || { mark "GEN FAIL $run $s"; return 1; }
  cp data/rval_greedy.parquet data/prec_g.parquet
  cp data/rval_sampled.parquet data/prec_s.parquet
  done_already $run $s greedy || { roll data/prec_g.parquet 4 data/prec_g_out.parquet && record $run $s greedy data/prec_g_out.parquet; }
  wait_idle || return 1
  done_already $run $s sampled || { roll data/prec_s.parquet 2 data/prec_s_out.parquet && record $run $s sampled data/prec_s_out.parquet; }
}

mark "filler start"
while true; do
  git pull -q --rebase 2>/dev/null || true
  # priority 1: v7b probe points already rolled by rval (extra reps)
  for s in $(grep -o '"step": [0-9]*' results/analysis/v7b_rollout_curve.jsonl 2>/dev/null | grep -o '[0-9]*' | sort -un); do
    done_already v7b $s greedy && done_already v7b $s sampled && continue
    mark "precision job: v7b step $s"
    job_v7b_probe $s || true
  done
  # priority 2: v7a pivots from local adapters
  for spec in "v7a 140 /workspace/v7_adapters/step_0140" "v7a 120 /workspace/v7_adapters/step_0120" "v7a 160 /workspace/v7_adapters/step_0160"; do
    set -- $spec
    done_already $1 $2 greedy && done_already $1 $2 sampled && continue
    mark "precision job: $1 step $2 (from adapter)"
    job_adapter $1 $2 $3 || true
  done
  # push any new records
  git add $OUT 2>/dev/null && git commit -q -m "precision re-rolls [pod]" 2>/dev/null && git pull -q --rebase 2>/dev/null && git push -q 2>/dev/null
  sleep 600
done
