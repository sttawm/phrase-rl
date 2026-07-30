#!/usr/bin/env python3
"""Sealed-leg phrase generation for an RL checkpoint (PREREG_closing_leg arm A/B).

Generates one greedy phrase per sealed task under a condition:
  polish: input = nominal,          tag "[input: original wording]"
  repair: input = ert_instruction,  tag "[input: adversarially reworded]"
Trace: sealed_assets_gemini.trace (row-17 convention). Prints every prompt head
and emitted phrase for PREFLIGHT INSPECTION (prereg requirement) — rolling only
happens after a human-visible check.
  .venv-gen/bin/python scripts/gen_sealed_arm.py <adapter_dir> <polish|repair> <out.parquet>
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

adapter_dir, condition, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
TAG = {"polish": "[input: original wording]",
       "repair": "[input: adversarially reworded]"}[condition]
_TAG_RE = re.compile(r"^\s*(?:\[input:[^\]]*\]\s*)+")

proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
base = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).eval()

frames = pd.read_parquet("results/sealed/sealed_frames.parquet")
assets = pd.read_parquet("results/sealed/sealed_assets_gemini.parquet").set_index("task")

rows = []
for r in frames.itertuples():
    a = assets.loc[r.task]
    src = r.nominal if condition == "polish" else a.ert_instruction
    img = Image.open(io.BytesIO(r.image_png)).convert("RGB")
    msgs = build_single_phrase_prefix(f"{TAG} {src}", img, trace=a.trace)
    inp = apply_template(proc, msgs, continue_final_message=True).to(model.device)
    plen = inp["input_ids"].shape[1]
    with torch.no_grad():
        g = model.generate(**inp, do_sample=False, max_new_tokens=48)
    p = _TAG_RE.sub("", proc.decode(g[0][plen:], skip_special_tokens=True)
                    .strip().split("\n")[0]).strip()
    rows.append({"task": r.task, "arm": f"sealed_{condition}", "phrase": p, "instruction": p})
    # preflight lines: prompt head (tag + input visible) and the emitted phrase
    print(f"PREFLIGHT [{r.task}]")
    print(f"  input : {TAG} {src[:90]}")
    print(f"  phrase: {p!r}", flush=True)

pd.DataFrame(rows).to_parquet(out_path, index=False)
print(f"-> {len(rows)} phrases -> {out_path}")
