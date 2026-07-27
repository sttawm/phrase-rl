#!/usr/bin/env python3
"""Generate greedy + sampled deployment phrases from a v7 checkpoint adapter for
the 4 fixed probe contexts. Mirrors phase2_train.run_probes generation exactly
(same build_single_phrase_prefix, greedy = do_sample=False). Writes
data/rval_greedy.parquet + data/rval_sampled.parquet (the rval roller's schema).

  gen_ckpt_phrases.py <adapter_dir> <n_samples>
"""
import io
import os
import sys

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, "src")
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from phrase_rl.cover_prompt import build_single_phrase_prefix
from phrase_rl.phase2_train import apply_template

adapter_dir = sys.argv[1]
n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 2

proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
base = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).eval()

ctx = pd.read_parquet(os.environ.get("PROBE_CONTEXTS", "results/phrase_artifacts/contexts_0c_tasks.parquet"))
tr = pd.read_parquet(os.environ.get("PROBE_TRACES", "results/phrase_artifacts/traces_0c_tasks.parquet"))
tmap = {(r.episode_index, r.t): r.trace for r in tr.itertuples()}


def dec(o, plen):
    return proc.decode(o[plen:], skip_special_tokens=True).strip().split("\n")[0].strip()


greedy, sampled = [], []
for r in ctx.itertuples():
    img = Image.open(io.BytesIO(r.image_png)).convert("RGB")
    msgs = build_single_phrase_prefix(r.instruction, img, trace=tmap.get((r.episode_index, r.t)))
    inp = apply_template(proc, msgs, continue_final_message=True).to(model.device)
    plen = inp["input_ids"].shape[1]
    with torch.no_grad():
        g = model.generate(**inp, do_sample=False, max_new_tokens=48)
    gp = dec(g[0], plen)
    greedy.append({"task": r.task, "arm": "v7_greedy", "phrase": gp, "instruction": gp})
    with torch.no_grad():
        ss = model.generate(**inp, do_sample=True, temperature=1.0,
                            num_return_sequences=max(4, n_samples), max_new_tokens=48)
    seen = []
    for o in ss:
        sp = dec(o, plen)
        if sp and sp not in seen:
            seen.append(sp)
    for sp in seen[:n_samples]:
        sampled.append({"task": r.task, "arm": "v7_sampled", "phrase": sp, "instruction": sp})

os.makedirs("data", exist_ok=True)
pd.DataFrame(greedy).to_parquet("data/rval_greedy.parquet", index=False)
pd.DataFrame(sampled).to_parquet("data/rval_sampled.parquet", index=False)
print(f"generated {len(greedy)} greedy, {len(sampled)} sampled from {adapter_dir}")
