#!/usr/bin/env bash
# Sealed-test leg runner: rolls ONE arm's phrases parquet on the sealed tasks.
#   run_sealed_leg.sh <arm_name> <phrases_parquet> <repeats>
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}" FINAL_EVAL=1
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
mark() { echo "[sealed $(date +%H:%M:%S)] $*" | tee -a /workspace/sealed.log; }
ARM=$1; PH=$2; REPS=$3; NW=${NW:-3}
OUT="results/sealed/${ARM}_x${REPS}.parquet"
mkdir -p results/sealed
[ -f "results/sealed/.done_${ARM}_x${REPS}" ] && { mark "skip $ARM (done)"; exit 0; }
PH=$PH NW=$NW .venv-gen/bin/python -c "
import os
import pandas as pd
d = pd.read_parquet(os.environ['PH'])
tasks = sorted(d.task.unique())
nw = int(os.environ['NW'])
for w in range(nw):
    wt = [t for i, t in enumerate(tasks) if i % nw == w]
    d[d.task.isin(wt)].to_parquet(f'data/sealed_w{w}.parquet', index=False)
print(f'{len(d)} phrase-rows over {len(tasks)} tasks')" || { mark "SHARD FAILED"; exit 1; }
rm -f data/sealed_out_w*.parquet
cd /workspace/INT-ACT
pids=""
for w in $(seq 0 $((NW-1))); do
  sleep 45
  FINAL_EVAL=1 /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
    --int-act-root /workspace/INT-ACT \
    --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
    --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
    --phrases /workspace/phrase-rl/data/sealed_w$w.parquet \
    --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
    --repeats $REPS \
    --out /workspace/phrase-rl/data/sealed_out_w$w.parquet > /workspace/sealed_w$w.log 2>&1 &
  pids="$pids $!"
done
wfail=0; for p in $pids; do wait $p || wfail=1; done
cd /workspace/phrase-rl
[ $wfail = 1 ] && { mark "WORKER FAILED $ARM"; exit 1; }
OUT=$OUT PH=$PH REPS=$REPS .venv-gen/bin/python -c "
import glob, os
import pandas as pd
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('data/sealed_out_w*.parquet'))], ignore_index=True)
ph = pd.read_parquet(os.environ['PH'])
expect = len(ph) * 24 * int(os.environ['REPS'])
assert len(d) == expect, f'expected {expect}, got {len(d)}'
d.to_parquet(os.environ['OUT'], index=False)
print(f'[sealed] RESULT {os.environ[\"OUT\"]}: pooled {d.success.mean()*100:.1f}%')" || { mark "MERGE FAILED $ARM"; exit 1; }
touch "results/sealed/.done_${ARM}_x${REPS}"
git add results/sealed && git commit -q -m "sealed: ${ARM} x${REPS} [pod]" && git pull -q --rebase && git push -q
mark "RESULT $ARM -> $OUT"
