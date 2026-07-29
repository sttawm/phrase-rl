#!/usr/bin/env python3
"""Reward-guided phrase search over the training corpus, candidate-gen stage
(user 2026-07-29: oracle-search-style exploration of good phrases + which
phrase-CHANGES help).

For the N biggest-club training instructions: sample K rewrites from the current
v7f policy (tagged nominal input, temp 1.0, deduped) + keep the original.
Output: results/analysis/search_candidates.parquet (instruction, phrase, source).
Contexts for gen: one club frame per instruction (the policy is image-conditioned).
Run on a pod with .venv-gen + the v7f adapter extracted (pod5).
  .venv-gen/bin/python scripts/build_search_candidates.py <adapter_dir> [N] [K]
"""
import io
import re
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
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
K = int(sys.argv[3]) if len(sys.argv) > 3 else 8
TAG = "[input: original wording]"
_TAG_RE = re.compile(r"^\s*(?:\[input:[^\]]*\]\s*)+")

proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
base = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).eval()

ctx = pd.read_parquet("data/contexts_club.parquet")
sizes = ctx.groupby("instruction").episode_index.nunique().sort_values(ascending=False)
targets = list(sizes.head(N).index)
print(f"{len(targets)} instructions (club sizes {sizes.iloc[0]}..{sizes.iloc[len(targets)-1]})")

rows = []
for i, ins in enumerate(targets):
    sub = ctx[ctx.instruction == ins].iloc[0]
    img = Image.open(io.BytesIO(sub.image_png)).convert("RGB")
    msgs = build_single_phrase_prefix(f"{TAG} {ins}", img, trace=None)
    inp = apply_template(proc, msgs, continue_final_message=True).to(model.device)
    plen = inp["input_ids"].shape[1]
    with torch.no_grad():
        ss = model.generate(**inp, do_sample=True, temperature=1.0,
                            num_return_sequences=K + 4, max_new_tokens=48)
    seen = {ins.strip().lower()}
    rows.append({"instruction": ins, "phrase": ins, "source": "original"})
    kept = 0
    for o in ss:
        p = _TAG_RE.sub("", proc.decode(o[plen:], skip_special_tokens=True)
                        .strip().split("\n")[0]).strip()
        if p and len(p.split()) >= 3 and p.lower() not in seen and kept < K:
            seen.add(p.lower())
            rows.append({"instruction": ins, "phrase": p, "source": "v7f_sample"})
            kept += 1
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(targets)}] {len(rows)} candidates", flush=True)
        pd.DataFrame(rows).to_parquet("results/analysis/search_candidates.parquet", index=False)

pd.DataFrame(rows).to_parquet("results/analysis/search_candidates.parquet", index=False)
print(f"-> {len(rows)} candidate rows (originals + samples)")
