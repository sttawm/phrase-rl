#!/usr/bin/env bash
# NOMINAL-INPUT x12 rollout eval — the reward-validity test: does the nominal
# greedy-reward gain translate into real rollout success on nominal inputs?
# Usage: run_nominal_eval12.sh <ARM_LABEL> <ADAPTER_DIR_OR_none> [workers]
#   ARM_LABEL: tag for the output file (e.g. B_s260, frozen, passthrough)
#   ADAPTER:   /workspace/adapters/B/step_0260 | none (frozen Qwen) | passthrough
# Output: results/val_screens/eval12nom_<LABEL>.parquet (2304 eps: 8x24x12)
# no xtrace: this script evals secrets from ~/.bashrc; -x would print them
# into tee'd logs (it did, once)
set -euo pipefail
LABEL="${1:?usage: run_nominal_eval12.sh <label> <adapter|none|passthrough> [workers]}"
ADAPTER="${2:?}"
NW="${3:-3}"
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline HF_HUB_DISABLE_XET=1
cd /workspace/phrase-rl
git pull --no-edit -q || true
# heal screen inputs from the git-tracked canonicals (MFS has eaten data/
# before; fresh pods have no data/ at all)
mkdir -p data
cp -f results/phrase_artifacts/val_screen_frames.parquet data/
test -f data/val_screen_frames.parquet
test -f results/phrase_artifacts/nominal_eval_assets.parquet

if [ "$ADAPTER" = "passthrough" ]; then
  # phrases = the nominal instructions themselves; TASKS_KEEP (csv) optionally
  # restricts to tasks lacking trustworthy historical passthrough numbers
  TASKS_KEEP="${TASKS_KEEP:-}" .venv-gen/bin/python - <<'PY'
import os
import pandas as pd
a = pd.read_parquet("results/phrase_artifacts/nominal_eval_assets.parquet")
keep = [t for t in os.environ.get("TASKS_KEEP", "").split(",") if t]
if keep:
    a = a[a.task.isin(keep)]
pd.DataFrame({"task": a.task, "arm": "tuned", "phrase": a.ert_instruction}).to_parquet(
    "data/nominal_phrases.parquet", index=False)
print("passthrough phrases:", len(a))
PY
else
  # phase4 requires --adapter and always generates BOTH arms; for "none"
  # (frozen Qwen) we pass any staged adapter and keep the base-arm rows
  KEEP_ARM=tuned
  GEN_ARGS="--adapter $ADAPTER"
  if [ "$ADAPTER" = "none" ]; then
    KEEP_ARM=base
    GEN_ARGS="--adapter $(ls -d /workspace/adapters/*_nom /workspace/adapters/*/step_* 2>/dev/null | head -1)"
  fi
  .venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
    --tasks data/val_screen_frames.parquet --ctx data/val_screen_frames.parquet \
    --assets results/phrase_artifacts/nominal_eval_assets.parquet $GEN_ARGS \
    --out data/nominal_phrases_all.parquet
  KEEP_ARM=$KEEP_ARM .venv-gen/bin/python - <<'PY'
import os
import pandas as pd
d = pd.read_parquet("data/nominal_phrases_all.parquet")
d = d[d.arm == os.environ["KEEP_ARM"]]
d.to_parquet("data/nominal_phrases.parquet", index=False)
print("greedy phrases:", len(d))
for r in d.itertuples():
    print(" ", r.task, "->", str(r.phrase)[:70])
PY
fi

.venv-gen/bin/python - <<PY
import pandas as pd
d = pd.read_parquet("data/nominal_phrases.parquet")
tasks = sorted(d.task.unique())
for w in range($NW):
    wt = [t for i, t in enumerate(tasks) if i % $NW == w]
    d[d.task.isin(wt)].to_parquet(f"data/nom_w{w}.parquet", index=False)
print("sharded across $NW workers")
PY
rm -f data/nom_out_w*.parquet
cd /workspace/INT-ACT
pids=""
for w in $(seq 0 $((NW-1))); do
  sleep 45
  /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
    --int-act-root /workspace/INT-ACT \
    --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
    --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
    --phrases /workspace/phrase-rl/data/nom_w$w.parquet \
    --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
    --repeats 12 \
    --out /workspace/phrase-rl/data/nom_out_w$w.parquet > /workspace/nom_w$w.log 2>&1 &
  pids="$pids $!"
done
wfail=0
for p in $pids; do wait $p || wfail=1; done
[ $wfail = 1 ] && { echo "NOMINAL-EVAL WORKER FAILED ($LABEL)"; exit 1; }
cd /workspace/phrase-rl
OUT="results/val_screens/eval12nom_${LABEL}.parquet" .venv-gen/bin/python - <<'PY'
import glob
import os
import pandas as pd
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/nom_out_w*.parquet"))], ignore_index=True)
import glob as _g
n_tasks = pd.read_parquet("data/nominal_phrases.parquet").task.nunique()
assert len(d) == 288 * n_tasks, f"expected {288*n_tasks}, got {len(d)}"
out = os.environ["OUT"]
d.to_parquet(out, index=False)
s = d.groupby("task").success.mean() * 100
print(f"[nominal-eval] RESULT {out}: pooled {d.success.mean()*100:.1f}% |", s.round(1).to_dict(), flush=True)
PY
git add results/val_screens/ && git -c user.name=nomeval -c user.email=pod@runpod commit -q -m "nominal-input x12: $LABEL [pod]" && git -c user.name=nomeval -c user.email=pod@runpod pull -q --rebase --no-edit && git push -q
echo "NOMINAL-EVAL-DONE ($LABEL)"
