#!/usr/bin/env python3
"""Row 17 phrase generation: nominal + gem-trace => Gemini-pro + RULES.

Identical to row 12's generation (gemini-pro-latest, thinking_budget=16384,
temperature=0.2, same prompt template) except the instruction slot carries the
NOMINAL instead of the ERT rewording. GEMINI_API_KEY must be in the env.
FINAL_EVAL=1 required (sealed-asset read).
"""
import os
import time

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

from google import genai
from google.genai import types

RULES = open("results/analysis/b4_phrasing_rules_v3.md").read()
ga = pd.read_parquet("results/sealed/sealed_assets_gemini.parquet")
client = genai.Client()

rows = []
for r in ga.itertuples():
    prompt = (f"{RULES}\n\n---\n\nApply the rules above.\n\nTrace:\n{r.trace}\n\n"
              f"Incoming instruction: {r.nominal}\n\n"
              f"Reply with ONLY the rewritten instruction.")
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model="gemini-pro-latest",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=20000,  # includes thinking tokens (16384) — 100 truncated to fragments, see ledger
                    thinking_config=types.ThinkingConfig(thinking_budget=16384),
                ),
            )
            p = (resp.text or "").strip().strip('"').split("\n")[0].strip()
            if len(p.split()) < 3:
                raise ValueError(f"fragment output: {p!r}")
            break
        except Exception as e:
            print(f"[retry {attempt}] {r.task}: {type(e).__name__}", flush=True)
            time.sleep(15 * (attempt + 1))
    else:
        raise SystemExit(f"generation failed for {r.task}")
    p = p or str(r.nominal)
    rows.append({"task": r.task, "arm": "rules_pro_nominal", "phrase": p,
                 "instruction": str(r.nominal)})
    same = "PASS-THROUGH" if p == str(r.nominal) else "rewrite"
    print(f"[row17] {r.task}: {p!r} ({same})", flush=True)

out = "results/sealed/ph_sealed_rules_pro_nominal.parquet"
pd.DataFrame(rows).to_parquet(out, index=False)
print(f"wrote {len(rows)} -> {out}")
