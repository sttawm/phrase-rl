#!/usr/bin/env python3
"""Amendment 20: Gemini + rules-v4 applied to the K=16 natural rephrases.

Each of the 192 rephrase16 phrases is rewritten by gemini-pro-latest under
b4_phrasing_rules_v3.md, using the SAME family wrap as the sealed ladder arms
(rules doc + sealed Gemini trace + incoming instruction; text-only, no image),
greedy. Output arm: rephrase16_rulesv4_gemini. Preflight-prints every line.
FINAL_EVAL=1 required. Writes results/sealed/ph_rr16_rulesv3_gemini.parquet
(task, arm, phrase, k, src).
"""
import os
import time

import pandas as pd
from google import genai
from google.genai import types

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

MODEL = "gemini-pro-latest"
RULES = open("results/analysis/b4_phrasing_rules_v3.md").read()
WRAP = """{rules}

---

Apply the rules above.

Trace:
{trace}

Incoming instruction: {src}

Reply with ONLY the rewritten instruction."""

client = genai.Client()
reph = pd.read_parquet("results/sealed/ph_sealed_rephrase16.parquet")
assets = pd.read_parquet("results/sealed/sealed_assets_gemini.parquet").set_index("task")

rows = []
for r in reph.itertuples():
    trace = assets.loc[r.task].trace
    prompt = WRAP.format(rules=RULES, trace=trace, src=r.phrase)
    out = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=[prompt],
                config=types.GenerateContentConfig(temperature=0.0))
            out = resp.text.strip().strip('"').split("\n")[0].strip()
            break
        except Exception as e:
            print(f"RETRY {r.task} k={r.k}: {type(e).__name__}", flush=True)
            time.sleep(10 * (attempt + 1))
    if not out:
        raise SystemExit(f"API failed x3 for {r.task} k={r.k}")
    rows.append({"task": r.task, "arm": "rr16_rulesv3_gemini",
                 "phrase": out, "k": int(r.k), "src": r.phrase})
    print(f"PREFLIGHT [{r.task}] k={r.k:02d}: {r.phrase!r} -> {out!r}", flush=True)

pd.DataFrame(rows).to_parquet("results/sealed/ph_rr16_rulesv3_gemini.parquet", index=False)
print(f"ALL-RULESV3-GEMINI-DONE n={len(rows)}")
