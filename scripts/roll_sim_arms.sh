#!/usr/bin/env bash
# Roll out the sim-scene arms with REAL episodes. Standing rule (user, 2026-08-09):
# sim scenes are ALWAYS evaluated by rollout, never by the proxy.
#
#   ARM=natadv   REPS=36 bash scripts/roll_sim_arms.sh     # the 180 generated nat+adv
#   ARM=oracle_screen POOL=20 REPS=12 bash scripts/roll_sim_arms.sh
#   ARM=oracle_final  REPS=36 bash scripts/roll_sim_arms.sh
#
# Resumable: --out accumulates, so it is REMOVED first and progress is recovered
# from the published parquet (phase0c appends; a stale --out silently inflates n).
set -uo pipefail
export UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR="${VLA_DATA_DIR:-/workspace/vla_data}" VLA_LOG_DIR="${VLA_LOG_DIR:-/workspace/vla_log}" WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl
ARM="${ARM:?set ARM}"; REPS="${REPS:-36}"; SHARD="${SHARD:-0}"; OF="${OF:-1}"
OUT="results/analysis/sim_rollouts_${ARM}_${SHARD}of${OF}.parquet"
mark(){ echo "[simroll $(date -u +%H:%M)] $*"; }

for i in 1 2 3; do timeout 400 git -c rebase.autoStash=true pull -q --rebase origin main && break
  git rebase --abort 2>/dev/null; rm -rf .git/rebase-merge .git/rebase-apply; sleep 10; done
mark "at $(git rev-parse --short HEAD) | ARM=$ARM REPS=$REPS shard $SHARD/$OF"

# ---- build the phrase list for this arm, minus anything already rolled ----
.venv-gen/bin/python - "$ARM" "$SHARD" "$OF" "$OUT" <<'PY'
import sys, pandas as pd, numpy as np, os
arm, shard, of, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
b = pd.read_parquet("results/analysis/bank_to_score.parquet")
sim = b[b.task.astype(str).str.startswith("widowx_")].copy()
sim["arm"] = np.where(sim.source.astype(str).str.startswith("generated_natural"), "natural",
             np.where(sim.source.astype(str).str.startswith("generated_adversarial"), "adversarial", "fine_exam"))
if arm == "natadv":
    d = sim[sim.arm.isin(["natural", "adversarial"])]
else:                                   # oracle stages read their own pool file
    d = pd.read_parquet(f"results/analysis/sim_oracle_pool.parquet")
done = set()
if os.path.exists(out):
    p = pd.read_parquet(out)
    done = set(zip(p.task, p.phrase))
    print(f"resume: {len(done)} phrase-cells already rolled", flush=True)
d = d[~d.set_index(["task", "phrase"]).index.isin(done)]
tasks = sorted(d.task.unique())
mine = [t for i, t in enumerate(tasks) if i % of == shard]
d = d[d.task.isin(mine)]
d[["task", "arm", "phrase"]].to_parquet("data/sim_arm_phrases.parquet", index=False)
print(f"to roll: {len(d)} phrases over {d.task.nunique()} tasks", flush=True)
PY
n=$(.venv-gen/bin/python -c "import pandas as pd;print(len(pd.read_parquet('data/sim_arm_phrases.parquet')))")
[ "$n" = "0" ] && { mark "nothing outstanding"; exit 0; }

rm -f data/sim_roll_tmp.parquet          # phase0c APPENDS to --out; stale file inflates n
cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/sim_arm_phrases.parquet \
  --episode-ids $(seq 0 $((REPS - 1))) --repeats 1 \
  --out /workspace/phrase-rl/data/sim_roll_tmp.parquet
rc=$?
cd /workspace/phrase-rl
mark "rollout rc=$rc"
[ -f data/sim_roll_tmp.parquet ] || { mark "FATAL no output"; exit 1; }

.venv-gen/bin/python - "$OUT" <<'PY'
import sys, pandas as pd, os
out = sys.argv[1]
new = pd.read_parquet("data/sim_roll_tmp.parquet")
agg = (new.groupby(["task", "arm", "phrase"])
          .agg(gt_success=("success", lambda s: round(100 * s.mean(), 2)),
               gt_n=("success", "size")).reset_index())
if os.path.exists(out):
    agg = pd.concat([pd.read_parquet(out), agg], ignore_index=True).drop_duplicates(["task", "phrase"], keep="last")
agg.to_parquet(out, index=False)
print(f"{len(agg)} phrase-cells -> {out}", flush=True)
print(agg.groupby("arm").agg(n=("phrase", "size"), mean_success=("gt_success", "mean")).round(1).to_string())
PY
git add -f "$OUT"
git commit -q -m "sim rollouts: $ARM shard $SHARD/$OF (REPS=$REPS)" 2>/dev/null
for i in 1 2 3; do
  timeout 900 bash -c "git -c rebase.autoStash=true pull -q --rebase origin main && git push -q origin HEAD:main" && { mark "pushed"; break; }
  git rebase --abort 2>/dev/null; rm -rf .git/rebase-merge .git/rebase-apply; sleep 20
done
mark "SIM-ROLL-DONE $ARM shard $SHARD"
