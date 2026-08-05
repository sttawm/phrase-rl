#!/usr/bin/env python3
"""v10 step-0 sealed anchors (Amendment 21): frozen BASE Qwen (no adapter)
under PROMPT B — build_single_phrase_prefix_bare, text-only, greedy — both
conditions. These cells double as the prompt-B bare-baseline row and v10's
pre-registered step-0. Preflight-prints every phrase.
FINAL_EVAL=1 required. Writes results/sealed/ph_sealed_v10step0_{polish,repair}.parquet.
"""
import os
import re
import sys

import pandas as pd
import torch

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

sys.path.insert(0, "src")
from transformers import AutoModelForImageTextToText, AutoProcessor

from phrase_rl.cover_prompt import build_single_phrase_prefix_bare
from phrase_rl.phase2_train import apply_template

proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda").eval()

assets = pd.read_parquet("results/sealed/sealed_assets_gemini.parquet")

for cond in ("polish", "repair"):
    out_path = f"results/sealed/ph_sealed_v10step0_{cond}.parquet"
    if os.path.exists(out_path):
        print(f"skip {cond} (exists)")
        continue
    rows = []
    for r in assets.itertuples():
        src = str(r.nominal if cond == "polish" else r.ert_instruction)
        msgs = build_single_phrase_prefix_bare(src, trace=r.trace)
        inp = apply_template(proc, msgs, continue_final_message=True).to(model.device)
        plen = inp["input_ids"].shape[1]
        with torch.no_grad():
            g = model.generate(**inp, do_sample=False, max_new_tokens=48)
        p = proc.decode(g[0][plen:], skip_special_tokens=True).strip().strip('"').split("\n")[0].strip()
        p = re.sub(r"^\s*(?:\[input:[^\]]*\]\s*)+", "", p).strip() or src
        rows.append({"task": r.task, "arm": f"v10step0_{cond}", "phrase": p, "instruction": src})
        print(f"PREFLIGHT [{cond}] {r.task}: {p!r}", flush=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"wrote 12 -> {out_path}", flush=True)
print("ALL-V10-ANCHORS-DONE")
