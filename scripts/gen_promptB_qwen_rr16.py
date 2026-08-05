#!/usr/bin/env python3
"""Amendment 22: frozen base Qwen under PROMPT B over the 192 rephrase16
phrases, greedy -> results/sealed/ph_rr16_promptB_qwen.parquet.
(The sealed-ERT Qwen prompt-B cell is v10step0_repair — already queued.)
FINAL_EVAL=1 required. Runs on a GPU pod with .venv-gen.
"""
import os
import sys

import pandas as pd
import torch

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

sys.path.insert(0, "src")
from transformers import AutoModelForImageTextToText, AutoProcessor

from phrase_rl.cover_prompt import build_single_phrase_prefix_bare
from phrase_rl.phase2_train import apply_template

out_path = "results/sealed/ph_rr16_promptB_qwen.parquet"
if os.path.exists(out_path):
    print("exists; skip")
    raise SystemExit(0)

proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda").eval()

assets = pd.read_parquet("results/sealed/sealed_assets_gemini.parquet").set_index("task")
reph = pd.read_parquet("results/sealed/ph_sealed_rephrase16.parquet")

rows = []
for r in reph.itertuples():
    msgs = build_single_phrase_prefix_bare(str(r.phrase), trace=assets.loc[r.task].trace)
    inp = apply_template(proc, msgs, continue_final_message=True).to(model.device)
    plen = inp["input_ids"].shape[1]
    with torch.no_grad():
        g = model.generate(**inp, do_sample=False, max_new_tokens=48)
    p = proc.decode(g[0][plen:], skip_special_tokens=True).strip().strip('"').split("\n")[0].strip() or str(r.phrase)
    rows.append({"task": r.task, "arm": "rr16_promptB_qwen", "phrase": p, "k": int(r.k), "src": str(r.phrase)})
    print(f"PREFLIGHT [{r.task}] k={r.k:02d}: {p!r}", flush=True)
pd.DataFrame(rows).to_parquet(out_path, index=False)
print(f"ALL-PROMPTB-QWEN-DONE n={len(rows)}")
