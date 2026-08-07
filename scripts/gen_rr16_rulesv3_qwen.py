#!/usr/bin/env python3
"""Amendment 23: frozen base Qwen under rules-v4 (ladder wrap) over the 192
rephrase16 phrases, greedy -> results/sealed/ph_rr16_rulesv3_qwen.parquet.
FINAL_EVAL=1 required. GPU pod with .venv-gen."""
import os
import re

import pandas as pd
import torch

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

from transformers import AutoModelForImageTextToText, AutoProcessor

WRAP = """{rules}

---

Apply the rules above.

Trace:
{trace}

Incoming instruction: {src}

Reply with ONLY the rewritten instruction."""

out_path = "results/sealed/ph_rr16_rulesv3_qwen.parquet"
if os.path.exists(out_path):
    print("exists; skip")
    raise SystemExit(0)

V4 = open("results/analysis/b4_phrasing_rules_v3.md").read()
proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda").eval()

assets = pd.read_parquet("results/sealed/sealed_assets_gemini.parquet").set_index("task")
reph = pd.read_parquet("results/sealed/ph_sealed_rephrase16.parquet")

rows = []
for r in reph.itertuples():
    prompt = WRAP.format(rules=V4, trace=assets.loc[r.task].trace, src=str(r.phrase))
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = proc(text=[text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        g = model.generate(**inp, do_sample=False, max_new_tokens=64)
    p = proc.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().strip('"').split("\n")[0].strip()
    p = re.sub(r"^\s*(?:\[input:[^\]]*\]\s*)+", "", p).strip() or str(r.phrase)
    rows.append({"task": r.task, "arm": "rr16_rulesv3_qwen", "phrase": p, "k": int(r.k), "src": str(r.phrase)})
    print(f"PREFLIGHT [{r.task}] k={r.k:02d}: {p!r}", flush=True)
pd.DataFrame(rows).to_parquet(out_path, index=False)
print(f"ALL-RULESV3-QWEN-DONE n={len(rows)}")
