#!/usr/bin/env bash
# One-off (pod3, idle window): tighten the red-team baselines (user 2026-07-11).
#   1. base greedy: reps x3 -> x6 (SE +-2.9 -> +-2.0pp) — resume rolls only reps 3-5
#   2. base_random: CoVer's "pi0 w/ random" analog — base Qwen generates a 32-list
#      per ERT instruction (same traces), ONE random phrase sampled per episode
#      (seeded), x3 reps. Answers: is base-51% the distribution or greedy's mode?
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl
git pull --no-edit

# 1) base 32-lists for the ERT instructions (trace-conditioned, temp 0.8)
.venv-gen/bin/python - <<'PY'
import io, json
import numpy as np, pandas as pd, torch
from PIL import Image
from phrase_rl.cover_prompt import build_qwen_messages, parse_reworded
from phrase_rl.phase2_train import apply_template

assets = pd.read_parquet("results/phrase_artifacts/redteam_eval_assets.parquet")
ctx = pd.read_parquet("results/phrase_artifacts/contexts_0c_tasks.parquet")
from transformers import AutoModelForImageTextToText, AutoProcessor
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda")

rows = []
rng = np.random.default_rng(7)
for a in assets.itertuples():
    frame = ctx[ctx.task == a.task].iloc[0]
    img = Image.open(io.BytesIO(frame["image_png"]))
    # trace goes in via the with-trace template (same conditioning as the base greedy arm)
    from phrase_rl.cover_prompt import build_user_prompt_with_trace, load_system_prompt
    msgs = [{"role": "system", "content": [{"type": "text", "text": load_system_prompt()}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": build_user_prompt_with_trace(a.ert_instruction, a.trace, 32)}]}]
    inputs = apply_template(processor, msgs, add_generation_prompt=True).to(model.device)
    torch.manual_seed(11)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=True, temperature=0.8, max_new_tokens=1400)
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    pool = parse_reworded(text)
    assert len(pool) >= 12, f"{a.task}: only {len(pool)} parsed"
    print(a.task, "pool:", len(pool))
    for ep in range(25):
        rows.append({"task": a.task, "arm": "base_random",
                     "phrase": str(pool[rng.integers(len(pool))]).strip(),
                     "instruction": a.ert_instruction, "episode_id": ep})
pd.DataFrame(rows).to_parquet("data/phrases_rt_random.parquet", index=False)
print("wrote", len(rows), "rows")
PY

# 2) base greedy phrases only (for the x6 top-up)
.venv-gen/bin/python - <<'PY'
import pandas as pd
base = pd.read_parquet("results/overnight/raw/phrases_rt_f300_l100.parquet")
base[base.arm == "base"].to_parquet("data/phrases_rt_basegreedy.parquet", index=False)
print("base greedy rows:", (base.arm == "base").sum())
PY

# 3a) base greedy: fresh CRN-era rolls, all reps
OUT=/workspace/phrase-rl/data/rollouts_rt_baselines_crn.parquet
[ -f "$OUT" ] || cp /workspace/phrase-rl/data/rollouts_rt_f300_l100.parquet "$OUT"
cd /workspace/INT-ACT
RUN() { /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 \
  --out "$OUT" "$@"; }
RUN --phrases /workspace/phrase-rl/data/phrases_rt_basegreedy.parquet --repeats 4
# 3b) base_random: episode-pinned rows, x3 reps
RUN --phrases /workspace/phrase-rl/data/phrases_rt_random.parquet --repeats 4

# 4) publish + summary (base_random x6 not needed: cap at what completed)
cd /workspace/phrase-rl
cp -f "$OUT" results/overnight/raw/
git add results/overnight/raw/rollouts_rt_baselines_crn.parquet
git -c user.name=pod3 -c user.email=pod@runpod commit -m "red-team baselines: base x6 reps + base_random arm (CoVer random analog) [pod]" || true
git -c user.name=pod3 -c user.email=pod@runpod pull --rebase --no-edit || true
git push || true
.venv-gen/bin/python - <<'PY'
import pandas as pd
d = pd.read_parquet("data/rollouts_rt_baselines_crn.parquet")
print((d.groupby("arm").success.agg(["mean","count"])).to_string())
PY
echo BASELINES-DONE
