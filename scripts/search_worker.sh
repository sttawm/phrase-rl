#!/usr/bin/env bash
# LLM-guided phrase search worker: processes any results/search/*_phrases.json
# lacking a *_results.json — rolls each phrase on layouts 0-17 x2 (search split;
# 18-23 reserved for confirmation), pushes the scoreboard, repeats.
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
mark() { echo "[search $(date +%H:%M:%S)] $*" | tee -a /workspace/search.log; }
NW=${NW:-3}
while true; do
  git pull -q --rebase 2>/dev/null
  for f in results/search/*_phrases.json; do
    [ -e "$f" ] || continue
    OUT="${f%_phrases.json}_results.json"
    [ -f "$OUT" ] && continue
    mark "processing $f"
    F=$f .venv-gen/bin/python -c "
import json, os
import pandas as pd
j = json.load(open(os.environ['F']))
rows = [{'task': j['task'], 'arm': 'search', 'phrase': p, 'instruction': p} for p in j['phrases']]
d = pd.DataFrame(rows)
nw = $NW
for w in range(nw):
    d.iloc[w::nw].to_parquet(f'data/search_w{w}.parquet', index=False)
print(len(d))" || { mark "BUILD FAILED $f"; continue; }
    rm -f data/search_out_w*.parquet
    cd /workspace/INT-ACT
    pids=""
    for w in $(seq 0 $((NW-1))); do
      sleep 45
      /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
        --int-act-root /workspace/INT-ACT \
        --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
        --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
        --phrases /workspace/phrase-rl/data/search_w$w.parquet \
        --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 \
        --repeats 2 \
        --out /workspace/phrase-rl/data/search_out_w$w.parquet > /workspace/search_w$w.log 2>&1 &
      pids="$pids $!"
    done
    wfail=0; for p in $pids; do wait $p || wfail=1; done
    cd /workspace/phrase-rl
    [ $wfail = 1 ] && { mark "WORKER FAILED $f"; continue; }
    F=$f OUT=$OUT .venv-gen/bin/python -c "
import glob, json, os
import pandas as pd
d = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob('data/search_out_w*.parquet'))], ignore_index=True)
s = (d.groupby('phrase').success.agg(['mean','count'])*[100,1]).round(1)
res = {'task': d.task.iloc[0], 'n_per_phrase': int(s['count'].iloc[0]),
       'scoreboard': [{'phrase': p, 'success_pct': float(r['mean']), 'n': int(r['count'])}
                      for p, r in s.sort_values('mean', ascending=False).iterrows()]}
json.dump(res, open(os.environ['OUT'], 'w'), indent=1)
print('top:', res['scoreboard'][0])" || { mark "MERGE FAILED $f"; continue; }
    git add results/search && git commit -q -m "search: $(basename $OUT) [pod2]" && git pull -q --rebase && git push -q
    mark "RESULT $(basename $OUT)"
  done
  sleep 60
done
