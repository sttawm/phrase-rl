#!/usr/bin/env python3
"""Amendment 22: Gemini under PROMPT B (no rules, no image, single line), greedy.
Two input sets:
  (a) the 192 rephrase16 phrases  -> results/sealed/ph_rr16_promptB_gemini.parquet
  (b) the 12 sealed ERT inputs    -> results/sealed/ph_sealed_promptB_gemini_ert.parquet
Model: gemini-pro-latest (arm-consistency with every prior Gemini arm; alias
caveat ledgered — generation date recorded in the parquet meta column).
FINAL_EVAL=1 required. Preflight-prints every line.
"""
import datetime
import os
import sys
import time

import pandas as pd
from google import genai
from google.genai import types

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

sys.path.insert(0, "src")
from phrase_rl.cover_prompt import PROMPT_B_SYSTEM, PROMPT_B_USER

MODEL = "gemini-pro-latest"
STAMP = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes")
client = genai.Client()
assets = pd.read_parquet("results/sealed/sealed_assets_gemini.parquet").set_index("task")


def rewrite(task, src):
    trace = assets.loc[task].trace
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=[PROMPT_B_USER.format(trace=trace, src=src)],
                config=types.GenerateContentConfig(system_instruction=PROMPT_B_SYSTEM, temperature=0.0))
            out = resp.text.strip().strip('"').split("\n")[0].strip()
            if out:
                return out
        except Exception as e:
            print(f"RETRY {task}: {type(e).__name__}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f"API failed x3 for {task}")


reph = pd.read_parquet("results/sealed/ph_sealed_rephrase16.parquet")
out_a = "results/sealed/ph_rr16_promptB_gemini.parquet"
if not os.path.exists(out_a):
    rows = []
    for r in reph.itertuples():
        p = rewrite(r.task, r.phrase)
        rows.append({"task": r.task, "arm": "rr16_promptB_gemini", "phrase": p,
                     "k": int(r.k), "src": r.phrase, "gen_utc": STAMP})
        print(f"PREFLIGHT-A [{r.task}] k={r.k:02d}: {p!r}", flush=True)
    pd.DataFrame(rows).to_parquet(out_a, index=False)
    print(f"wrote {len(rows)} -> {out_a}", flush=True)

out_b = "results/sealed/ph_sealed_promptB_gemini_ert.parquet"
if not os.path.exists(out_b):
    rows = []
    for task, a in assets.iterrows():
        p = rewrite(task, str(a.ert_instruction))
        rows.append({"task": task, "arm": "promptB_gemini_ert", "phrase": p,
                     "instruction": str(a.ert_instruction), "gen_utc": STAMP})
        print(f"PREFLIGHT-B [{task}]: {p!r}", flush=True)
    pd.DataFrame(rows).to_parquet(out_b, index=False)
    print(f"wrote {len(rows)} -> {out_b}", flush=True)
print("ALL-PROMPTB-GEMINI-DONE")
