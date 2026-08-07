#!/usr/bin/env python3
"""v11 training-tier naturals, Qwen arm (A27): trace-conditioned, knowledge-free
natural rewordings of the 1,361 unique training instructions, K=3 each at
temperature 1.0. These are training INPUTS (the naturals the policy learns to
handle), not rule-following outputs — the prompt carries no phrasing knowledge.
Writes results/phrase_artifacts/naturals_qwen_v11.parquet (instruction, phrase, author).
"""
import re
import sys

import pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT = """You are helping reword instructions for a tabletop robot task.

Scene analysis (what is in the scene and what the instruction refers to):
{trace}

Original instruction: "{src}"

Write 3 different rewordings of this instruction, the way three different people might naturally say it when asking for the task to be done.
- Each rewording must keep the same objects and the same goal.
- Natural, everyday phrasing — the way real people talk.
- You may refer to objects the way the scene analysis describes them.
Output exactly 3 lines, one rewording per line, no numbering, no quotes."""

proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda").eval()

t = pd.read_parquet("results/phrase_artifacts/cover35_teacher_train.parquet")
u = t.drop_duplicates("instruction")[["instruction", "trace"]]
rows = []
for i, r in enumerate(u.itertuples()):
    msgs = [{"role": "user", "content": [{"type": "text",
             "text": PROMPT.format(trace=r.trace, src=r.instruction)}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = proc(text=[text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        g = model.generate(**inp, do_sample=True, temperature=1.0, max_new_tokens=96)
    out = proc.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    lines = [re.sub(r'^\s*\d+[.)]\s*', '', l).strip().strip('"') for l in out.split("\n") if l.strip()][:3]
    for p in lines:
        if p and p.lower() != r.instruction.lower():
            rows.append({"instruction": r.instruction, "phrase": p, "author": "qwen"})
    if i % 100 == 0:
        print(f"[{i}/{len(u)}] {r.instruction!r} -> {lines[:1]}", flush=True)

pd.DataFrame(rows).to_parquet("results/phrase_artifacts/naturals_qwen_v11.parquet", index=False)
print(f"NATURALS-QWEN-DONE n={len(rows)}")
