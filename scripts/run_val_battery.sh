#!/usr/bin/env bash
# VAL-tier red-team baseline battery (pod 3, gated behind idmirror).
# 8 native val tasks x 3 arms x 24 layouts x 12 reps (~6.9k eps):
#   redteam_direct : the ERT instruction passed through untouched
#   base_greedy    : frozen Qwen, single-phrase deployment prompt (ERT + trace)
#   base_random16  : frozen Qwen 16-list on same inputs, uniform pick (seed 0)
# ERT inputs: CoVer's own instructions for the 4 ID tasks (continuity with all
# prior red-team evals); valtier_ert_bank.json (v3, escalated) for the 4 val-tier
# tasks. CoVer-select arm + CoVer's 3 OOD tasks append later (CRN keys pair up).
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
while tmux has-session -t idmirror 2>/dev/null; do sleep 300; done
git pull --no-edit || true
test -f results/overnight/raw/rollouts_id_nominal.parquet  # idmirror must have finished

# 1) traces for the 4 val-tier ERT inputs (ID tasks already have them in redteam_eval_assets)
if [ ! -f data/valtier_ert_assets.parquet ]; then
  .venv-gen/bin/python -m phrase_rl.gemini_redteam_assets \
    --ert results/phrase_artifacts/valtier_ert_bank.json \
    --contexts data/valtask_frames.parquet \
    --out data/valtier_ert_assets.parquet
fi

# 2) frozen-Qwen phrase arms: greedy single + random-of-16 per task
if [ ! -f data/battery_phrases.parquet ]; then
  .venv-gen/bin/python - <<'PY'
import io
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from phrase_rl import cover_prompt
from phrase_rl.phase2_train import apply_template

id_assets = pd.read_parquet("results/phrase_artifacts/redteam_eval_assets.parquet")
vt_assets = pd.read_parquet("data/valtier_ert_assets.parquet")
id_ctx = pd.read_parquet("data/contexts_0c_tasks.parquet")
vt_ctx = pd.read_parquet("data/valtask_frames.parquet")
frames = {r.task: r.image_png for d in (id_ctx, vt_ctx) for r in d.itertuples()}
assets = pd.concat([id_assets[["task","ert_instruction","trace"]],
                    vt_assets[["task","ert_instruction","trace"]]], ignore_index=True)

proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda")
rng = np.random.default_rng(0)
rows = []
for a in assets.itertuples():
    img = Image.open(io.BytesIO(frames[a.task]))
    rows.append({"task": a.task, "arm": "redteam_direct", "phrase": str(a.ert_instruction)})
    # greedy single-phrase (deployment prompt)
    msgs = cover_prompt.build_single_phrase_prefix(str(a.ert_instruction), img, trace=str(a.trace))
    inp = apply_template(proc, msgs, continue_final_message=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, do_sample=False, max_new_tokens=48)
    g = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip()
    rows.append({"task": a.task, "arm": "base_greedy", "phrase": g})
    # 16-list, uniform random pick
    msgs = cover_prompt.build_qwen_messages(image=img, instruction=str(a.ert_instruction),
                                            batch_number=16, trace=str(a.trace))
    inp = apply_template(proc, msgs, add_generation_prompt=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, do_sample=True, temperature=0.8, max_new_tokens=900)
    text = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    cands = cover_prompt.parse_reworded(text)
    assert len(cands) >= 8, f"{a.task}: only {len(cands)} candidates parsed"
    rows.append({"task": a.task, "arm": "base_random16", "phrase": str(cands[rng.integers(len(cands))])})
    print(a.task, "done")
pd.DataFrame(rows).to_parquet("data/battery_phrases.parquet", index=False)
print(len(rows), "battery phrase rows")
PY
fi

# 3) rollouts x12, CRN
cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/battery_phrases.parquet \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --repeats 12 \
  --out /workspace/phrase-rl/data/rollouts_val_battery.parquet

cd /workspace/phrase-rl
cp -f data/rollouts_val_battery.parquet data/battery_phrases.parquet data/valtier_ert_assets.parquet results/overnight/raw/
git add results/overnight/raw/ \
  && git -c user.name=pod3 -c user.email=pod@runpod commit -m "VAL battery: redteam_direct/base_greedy/base_random16 x 8 tasks x12 [pod]" \
  && git -c user.name=pod3 -c user.email=pod@runpod pull --rebase --no-edit \
  && git push
echo VAL-BATTERY-DONE
