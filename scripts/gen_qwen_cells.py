#!/usr/bin/env python3
"""Four Qwen sealed-cell generations in one model load (Amendments 15/16):
  1. qwen_bare_ert           — unified bare template, ERT input, no thinking
  2. rules_v3_qwen_nominal   — rules-v3 doc, nominal input, no thinking
  3. rules_v4_qwen_think_ert     — rules-v4 doc, ERT input, thinking ON
  4. rules_v4_qwen_think_nominal — rules-v4 doc, nominal input, thinking ON
Text + full Gemini trace only (no image), greedy, one call per task.
FINAL_EVAL=1 required. Run on a pod with .venv-gen (pod5)."""
import os
import re
import sys

import pandas as pd
import torch

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

from transformers import AutoModelForImageTextToText, AutoProcessor

BARE = """You rewrite one tabletop-manipulation instruction for a frozen pi0 robot policy trained on the Bridge corpus. How well the policy performs depends on the exact wording of the instruction it is given.

**You receive:**
1. An instruction.
2. A scene trace — a scene description plus a mapping from each instruction phrase to a plain object name.

**You output:** exactly ONE rewritten instruction. One line, lowercase, no final period, no quotes, no commentary. Reason internally; reply with the line alone.

---

Rewrite the instruction that follows using your best judgment.

Trace:
{trace}

Incoming instruction: {src}

Reply with ONLY the rewritten instruction."""

RULES_WRAP = """{rules}

---

Apply the rules above.

Trace:
{trace}

Incoming instruction: {src}

Reply with ONLY the rewritten instruction."""

V3 = open("results/analysis/b4_phrasing_rules_v3.md").read()
V4 = open("results/analysis/b4_phrasing_rules_v4.md").read()
ga = pd.read_parquet("results/sealed/sealed_assets_gemini.parquet")

proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda").eval()

CELLS = [
    ("qwen_bare_ert",            "ert",     None, False),
    ("rules_v3_qwen_nominal",    "nominal", V3,   False),
    ("rules_v4_qwen_think_ert",  "ert",     V4,   True),
    ("rules_v4_qwen_think_nominal", "nominal", V4, True),
]

for arm, cond, rules, think in CELLS:
    out_path = f"results/sealed/ph_sealed_{arm}.parquet"
    if os.path.exists(out_path):
        print(f"skip {arm} (exists)")
        continue
    rows = []
    for r in ga.itertuples():
        src = str(r.nominal if cond == "nominal" else r.ert_instruction)
        if rules is None:
            prompt = BARE.format(trace=r.trace, src=src)
        else:
            prompt = RULES_WRAP.format(rules=rules, trace=r.trace, src=src)
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=think)
        inp = proc(text=[text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**inp, do_sample=False,
                               max_new_tokens=14336 if think else 64)
        dec = proc.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        if think:
            print(f"RAW-HEAD [{arm}] {r.task}: {dec[:300]!r}", flush=True)
            print(f"RAW-TAIL: {dec[-200:]!r}", flush=True)
        if think:
            if "</think>" in dec:
                dec = dec.split("</think>")[-1]          # reasoned, closed: take the answer
            elif dec.lstrip().startswith("<think>"):
                print(f"PARSE-FAIL(truncated think) [{arm}] {r.task} — len {len(dec)}", flush=True)
                dec = ""                                  # true truncation -> loud fallback
            # else: model answered plainly without think markup — dec IS the answer
        p = dec.strip().strip('"').split("\n")[0].strip()
        p = re.sub(r"^\s*(?:\[input:[^\]]*\]\s*)+", "", p).strip()
        if not p or len(p.split()) < 3:
            p = src  # degenerate output falls back to passthrough; ledger via preflight
        rows.append({"task": r.task, "arm": arm, "phrase": p, "instruction": src})
        print(f"PREFLIGHT [{arm}] {r.task}: {p!r}", flush=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"wrote 12 -> {out_path}", flush=True)
print("ALL-QWEN-CELLS-DONE")
