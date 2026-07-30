#!/usr/bin/env python3
"""Arm E phrase generation (PREREG Amendment 7): rules-v4 + gemini-pro on sealed.

Provenance-matched to rows 12/17 (gen_row17_nominal_rules.py): gemini-pro-latest,
temperature=0.2, thinking_budget=16384, sealed gemini traces. Condition arg picks
the instruction slot: nominal | ert. Prints every phrase (preflight required
before rolling). GEMINI_API_KEY in env; FINAL_EVAL=1 required.
  .venv/bin/python scripts/gen_rules_v4_sealed.py nominal
  .venv/bin/python scripts/gen_rules_v4_sealed.py ert
"""
import os
import sys
import time

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")
condition = sys.argv[1]
assert condition in ("nominal", "ert")

from google import genai
from google.genai import types

RULES = open("results/analysis/b4_phrasing_rules_v4.md").read()
ga = pd.read_parquet("results/sealed/sealed_assets_gemini.parquet")
client = genai.Client()

rows = []
for r in ga.itertuples():
    src = r.nominal if condition == "nominal" else r.ert_instruction
    prompt = (f"{RULES}\n\n---\n\nApply the rules above.\n\nTrace:\n{r.trace}\n\n"
              f"Incoming instruction: {src}\n\n"
              f"Reply with ONLY the rewritten instruction.")
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model="gemini-pro-latest",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=20000,
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
    rows.append({"task": r.task, "arm": f"rules_v4_{condition}", "phrase": p,
                 "instruction": str(src)})
    print(f"PREFLIGHT [{r.task}] ({condition})")
    print(f"  input : {str(src)[:100]}")
    print(f"  phrase: {p!r}", flush=True)

out = f"results/sealed/ph_sealed_rules_v4_{condition}.parquet"
pd.DataFrame(rows).to_parquet(out, index=False)
print(f"wrote {len(rows)} -> {out}")
