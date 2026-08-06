#!/usr/bin/env python3
"""val-8 reference gen on pod6: Qwen + rules-v4 on the 8 adversarial probe
inputs (ref 903). Mirrors gen_rules_qwen_rr16.py exactly (chat template,
greedy, thinking off, tag-strip); ERT-derived probe traces."""
import re

import pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

WRAP = """{rules}

---

Apply the rules above.

Trace:
{trace}

Incoming instruction: {src}

Reply with ONLY the rewritten instruction."""

V4 = open("results/analysis/b4_phrasing_rules_v4.md").read()
proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda").eval()

ctx = pd.read_parquet("results/phrase_artifacts/contexts_probe8_adv.parquet")
tr = pd.read_parquet("results/phrase_artifacts/traces_probe8_adv.parquet")
tmap = {(r.episode_index, r.t): r.trace for r in tr.itertuples()}

rows = []
for r in ctx.itertuples():
    prompt = WRAP.format(rules=V4, trace=tmap[(r.episode_index, r.t)], src=r.instruction)
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = proc(text=[text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        g = model.generate(**inp, do_sample=False, max_new_tokens=64)
    p = proc.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().strip('"').split("\n")[0].strip()
    p = re.sub(r"^\s*(?:\[input:[^\]]*\]\s*)+", "", p).strip() or str(r.instruction)
    rows.append({"task": r.task, "arm": "val8_rulesv4_qwen", "phrase": p, "instruction": r.instruction})
    print(f"PREFLIGHT [{r.task}]: {p!r}", flush=True)

pd.DataFrame(rows).to_parquet("results/phrase_artifacts/dev10q_g_v10ref_0903.parquet", index=False)
print(f"VAL8-RULES-QWEN-DONE n={len(rows)}")
