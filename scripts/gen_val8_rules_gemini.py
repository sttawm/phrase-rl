#!/usr/bin/env python3
"""val-8 reference: Gemini-Pro + rules-v4 applied to the 8 adversarial probe
inputs (same WRAP/temp-0 protocol as the sealed rules arms; ERT-derived traces).
Writes dev10q_g_v10ref_0901.parquet — consumed by the cell pool as ref 901."""
import os
import time

import pandas as pd
from google import genai
from google.genai import types

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

MODEL = "gemini-pro-latest"
RULES = open("results/analysis/b4_phrasing_rules_v4.md").read()
WRAP = """{rules}

---

Apply the rules above.

Trace:
{trace}

Incoming instruction: {src}

Reply with ONLY the rewritten instruction."""

client = genai.Client()
ctx = pd.read_parquet("results/phrase_artifacts/contexts_probe8_adv.parquet")
tr = pd.read_parquet("results/phrase_artifacts/traces_probe8_adv.parquet")
tmap = {(r.episode_index, r.t): r.trace for r in tr.itertuples()}

rows = []
for r in ctx.itertuples():
    trace = tmap[(r.episode_index, r.t)]
    prompt = WRAP.format(rules=RULES, trace=trace, src=r.instruction)
    out = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=[prompt],
                config=types.GenerateContentConfig(temperature=0.0))
            out = resp.text.strip().strip('"').split("\n")[0].strip()
            break
        except Exception as e:
            print(f"RETRY {r.task}: {type(e).__name__}", flush=True)
            time.sleep(10 * (attempt + 1))
    if not out:
        raise SystemExit(f"API failed x3 for {r.task}")
    rows.append({"task": r.task, "arm": "val8_rulesv4_gemini",
                 "phrase": out, "instruction": r.instruction})
    print(f"PREFLIGHT [{r.task}]: {r.instruction!r} -> {out!r}", flush=True)

pd.DataFrame(rows).to_parquet(
    "results/phrase_artifacts/dev10q_g_v10ref_0901.parquet", index=False)
print(f"VAL8-RULES-GEMINI-DONE n={len(rows)}")
