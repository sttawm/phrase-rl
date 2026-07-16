#!/usr/bin/env bash
# Battery PHRASE GENERATION only (pod 3): produces the 3 baseline arms' phrases
# for all 8 tasks and COMMITS them so rollout shards can pull and roll.
# Requires: valloop NOT running (Qwen VRAM), valtier_ert_assets present.
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
test -f data/valtier_ert_assets.parquet

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
    msgs = cover_prompt.build_single_phrase_prefix(str(a.ert_instruction), img, trace=str(a.trace))
    inp = apply_template(proc, msgs, continue_final_message=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, do_sample=False, max_new_tokens=48)
    g = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip()
    rows.append({"task": a.task, "arm": "base_greedy", "phrase": g})
    msgs = cover_prompt.build_qwen_messages(image=img, instruction=str(a.ert_instruction),
                                            batch_number=16, trace=str(a.trace))
    inp = apply_template(proc, msgs, add_generation_prompt=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, do_sample=True, temperature=0.8, max_new_tokens=900)
    text = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    cands = cover_prompt.parse_reworded(text)
    assert len(cands) >= 8, f"{a.task}: only {len(cands)} candidates"
    rows.append({"task": a.task, "arm": "base_random16", "phrase": str(cands[rng.integers(len(cands))])})
    print(a.task, "ok")
pd.DataFrame(rows).to_parquet("results/phrase_artifacts/battery_phrases.parquet", index=False)
print(len(rows), "battery phrase rows")
PY
git add results/phrase_artifacts/battery_phrases.parquet
git -c user.name=pod3 -c user.email=pod@runpod commit -m "battery phrases: 3 baseline arms x 8 tasks [pod]" || true
git -c user.name=pod3 -c user.email=pod@runpod pull --rebase --no-edit
git push
echo BATGEN-DONE
