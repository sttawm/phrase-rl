#!/usr/bin/env bash
# NIGHT SUITE (pod 3): task-generalization eval on the 4 designated VAL TASKS —
# tasks NO arm has ever trained on. Arms: original, base, rollout@s80, l2@280,
# flow@40 (adapters expected in results/checkpoints/eval_adapters/{rollout_s80,l2_280,flow_40}).
# Output: data/rollouts_valtasks.parquet (+ pushed). ~4,600 episodes.
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl
git pull --no-edit
mkdir -p data

# 0) top-phrase sampling on the ID tasks (12 draws/arm/task -> distribution modes)
if [ ! -f data/top_phrases.parquet ]; then
  cp -f results/phrase_artifacts/contexts_0c_tasks.parquet data/contexts_0c_tasks.parquet
  .venv-gen/bin/python -m phrase_rl.sample_top_phrases \
    --adapters base=NONE rollout_s80=results/checkpoints/eval_adapters/rollout_s80 \
               l2_280=results/checkpoints/eval_adapters/l2_280 flow_40=results/checkpoints/eval_adapters/flow_40 \
    --assets results/phrase_artifacts/redteam_eval_assets.parquet \
    --ctx data/contexts_0c_tasks.parquet --n 12 --out data/top_phrases.parquet
  cp -f data/top_phrases.parquet results/overnight/raw/
fi

VALTASKS="widowx_coke_can_on_plate_clean widowx_carrot_on_keyboard_clean widowx_coke_can_on_ramekin_clean widowx_carrot_on_wheel_clean"

# 1) frames + nominal instructions for the val tasks
if [ ! -f data/valtask_frames.parquet ]; then
  /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/sim_train_contexts.py \
    --int-act-root /workspace/INT-ACT --tasks $VALTASKS --episode-ids 0 \
    --out /workspace/phrase-rl/data/valtask_frames.parquet
fi

# 2) instruction json + Gemini traces (assets format reused from the redteam pipeline)
if [ ! -f data/valtask_assets.parquet ]; then
  .venv-gen/bin/python - <<'PY'
import json
import pandas as pd
d = pd.read_parquet("data/valtask_frames.parquet")
json.dump({r.task: str(r.instruction) for r in d.itertuples()}, open("data/valtask_instructions.json", "w"), indent=1)
print(open("data/valtask_instructions.json").read())
PY
  .venv-gen/bin/python -m phrase_rl.gemini_redteam_assets \
    --ert data/valtask_instructions.json --contexts data/valtask_frames.parquet \
    --out data/valtask_assets.parquet
fi

# 3) phrases per arm (base+tuned per adapter; original rides along)
GEN() { .venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
  --tasks data/valtask_frames.parquet --ctx data/valtask_frames.parquet \
  --assets data/valtask_assets.parquet --adapter "$1" --out "$2"; }
GEN results/checkpoints/eval_adapters/rollout_s80 data/vt_rollout.parquet
GEN results/checkpoints/eval_adapters/l2_280 data/vt_l2.parquet
GEN results/checkpoints/eval_adapters/flow_40 data/vt_flow.parquet

.venv-gen/bin/python - <<'PY'
import pandas as pd
ro = pd.read_parquet("data/vt_rollout.parquet")
l2 = pd.read_parquet("data/vt_l2.parquet")
fl = pd.read_parquet("data/vt_flow.parquet")
frames = [ro[ro.arm=="base"],                       # shared baseline (same weights every file)
          ro[ro.arm=="original"],
          ro[ro.arm=="tuned"].assign(arm="rollout_s80"),
          l2[l2.arm=="tuned"].assign(arm="tuned_l2_280"),
          fl[fl.arm=="tuned"].assign(arm="tuned_flow_40")]
out = pd.concat(frames, ignore_index=True).drop_duplicates(["task","arm"])
out.to_parquet("data/phrases_valtasks.parquet", index=False)
print(out[["task","arm","phrase"]].to_string())
PY

# 4) rollouts x12 on the val tasks' full grids
cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/phrases_valtasks.parquet \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --repeats 12 \
  --out /workspace/phrase-rl/data/rollouts_valtasks.parquet

# 5) publish + table
cd /workspace/phrase-rl
cp -f data/rollouts_valtasks.parquet data/phrases_valtasks.parquet data/valtask_assets.parquet results/overnight/raw/
git add results/overnight/raw/
git -c user.name=pod3 -c user.email=pod@runpod commit -m "TASK-GENERALIZATION suite: 5 arms x 4 never-trained val tasks x12 [pod]" || true
git -c user.name=pod3 -c user.email=pod@runpod pull --rebase --no-edit || true
git push || true
.venv-gen/bin/python - <<'PY'
import pandas as pd
d = pd.read_parquet("data/rollouts_valtasks.parquet")
print((d.pivot_table(index="task", columns="arm", values="success", aggfunc="mean")*100).round(1).to_string())
print("\noverall:"); print((d.groupby("arm").success.mean()*100).round(1).to_string())
PY
echo VALTASK-NIGHT-DONE
