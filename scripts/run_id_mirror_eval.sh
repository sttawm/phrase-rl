#!/usr/bin/env bash
# ID NOMINAL MIRROR (pod 3, gated behind the study queue): the missing eval cell.
# Same methodology as the val-task suite — 5 arms (original, base, rollout_s80,
# l2_280, flow_40), NOMINAL instruction inputs, x12 reps x 24 layouts — but on the
# 4 TRAINED (ID) tasks. Output: data/rollouts_id_nominal.parquet (+ pushed).
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl

# gate: wait for the study queue session to finish, then require its final artifact
while tmux has-session -t studyq 2>/dev/null; do sleep 120; done
git pull --no-edit || true
test -f data/rollouts_study_ert2_topup.parquet  # fail loud if studyq died early

# 1) nominal-instruction assets for the ID tasks (Gemini trace per task, ~8 calls)
if [ ! -f data/id_nominal_assets.parquet ]; then
  cp -f results/phrase_artifacts/contexts_0c_tasks.parquet data/contexts_0c_tasks.parquet
  .venv-gen/bin/python - <<'PY'
import json
import pandas as pd
d = pd.read_parquet("data/contexts_0c_tasks.parquet")
json.dump({r.task: str(r.instruction) for r in d.itertuples()},
          open("data/id_nominal_instructions.json", "w"), indent=1)
print(open("data/id_nominal_instructions.json").read())
PY
  .venv-gen/bin/python -m phrase_rl.gemini_redteam_assets \
    --ert data/id_nominal_instructions.json --contexts data/contexts_0c_tasks.parquet \
    --out data/id_nominal_assets.parquet
fi

# 2) phrases per arm (nominal input + trace; base rides along in each file)
GEN() { .venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
  --tasks data/contexts_0c_tasks.parquet --ctx data/contexts_0c_tasks.parquet \
  --assets data/id_nominal_assets.parquet --adapter "$1" --out "$2"; }
GEN results/checkpoints/eval_adapters/rollout_s80 data/idm_rollout.parquet
GEN results/checkpoints/eval_adapters/l2_280 data/idm_l2.parquet
GEN results/checkpoints/eval_adapters/flow_40 data/idm_flow.parquet

.venv-gen/bin/python - <<'PY'
import pandas as pd
ro = pd.read_parquet("data/idm_rollout.parquet")
l2 = pd.read_parquet("data/idm_l2.parquet")
fl = pd.read_parquet("data/idm_flow.parquet")
frames = [ro[ro.arm=="base"], ro[ro.arm=="original"],
          ro[ro.arm=="tuned"].assign(arm="rollout_s80"),
          l2[l2.arm=="tuned"].assign(arm="tuned_l2_280"),
          fl[fl.arm=="tuned"].assign(arm="tuned_flow_40")]
out = pd.concat(frames, ignore_index=True).drop_duplicates(["task","arm"])
out.to_parquet("data/phrases_id_nominal.parquet", index=False)
print(out[["task","arm","phrase"]].to_string())
PY

# 3) rollouts x12, full grids, CRN
cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/phrases_id_nominal.parquet \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --repeats 12 \
  --out /workspace/phrase-rl/data/rollouts_id_nominal.parquet

cd /workspace/phrase-rl
cp -f data/rollouts_id_nominal.parquet data/phrases_id_nominal.parquet data/id_nominal_assets.parquet results/overnight/raw/
git add results/overnight/raw/ \
  && git -c user.name=pod3 -c user.email=pod@runpod commit -m "ID NOMINAL MIRROR: 5 arms x 4 ID tasks x12, val-task-suite methodology [pod]" \
  && git -c user.name=pod3 -c user.email=pod@runpod pull --rebase --no-edit \
  && git push
.venv-gen/bin/python - <<'PY'
import pandas as pd
d = pd.read_parquet("data/rollouts_id_nominal.parquet")
print((d.pivot_table(index="task", columns="arm", values="success", aggfunc="mean")*100).round(1).to_string())
print("\noverall:"); print((d.groupby("arm").success.mean()*100).round(1).to_string())
PY
echo ID-MIRROR-DONE
