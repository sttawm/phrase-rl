#!/usr/bin/env python3
"""Resume the rules-v3 Gemini arm from a partial run's log.

The first attempt hung on an untimed API call at 150/192. Completed rewrites are
recoverable verbatim from the PREFLIGHT lines, so this parses those, generates
only the missing (task, k) cells — with a hard per-call timeout this time — and
writes the full parquet.

  GEMINI_API_KEY=... FINAL_EVAL=1 .venv/bin/python \
      scripts/gen_rr16_rulesv3_gemini_resume.py <logfile>
"""
import os
import re
import sys
import time

import pandas as pd
from google import genai
from google.genai import types

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

LOG = sys.argv[1]
OUT = "results/sealed/ph_rr16_rulesv3_gemini.parquet"
MODEL = "gemini-pro-latest"
RULES = open("results/analysis/b4_phrasing_rules_v3.md").read()
WRAP = """{rules}

---

Apply the rules above.

Trace:
{trace}

Incoming instruction: {src}

Reply with ONLY the rewritten instruction."""

# PREFLIGHT [task] k=NN: 'src' -> 'out'
PAT = re.compile(r"^PREFLIGHT \[(?P<task>[a-z0-9_]+)\] k=(?P<k>\d+): '(?P<src>.*)' -> '(?P<out>.*)'$")
done = {}
for line in open(LOG):
    m = PAT.match(line.rstrip("\n"))
    if m:
        done[(m.group("task"), int(m.group("k")))] = m.group("out")
print(f"recovered {len(done)} completed rewrites from log")

reph = pd.read_parquet("results/sealed/ph_sealed_rephrase16.parquet")
assets = pd.read_parquet("results/sealed/sealed_assets_gemini.parquet").set_index("task")
client = genai.Client()

rows, made = [], 0
for r in reph.itertuples():
    key = (r.task, int(r.k))
    if key in done:
        rows.append({"task": r.task, "k": int(r.k), "arm": "rr16_rulesv3_gemini",
                     "phrase": done[key], "instruction": r.phrase})
        continue
    prompt = WRAP.format(rules=RULES, trace=assets.loc[r.task].trace, src=r.phrase)
    out = None
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0, http_options=types.HttpOptions(timeout=90_000)))
            out = (resp.text or "").strip().strip('"').split("\n")[0].strip()
            if out:
                break
        except Exception as e:
            print(f"RETRY {r.task} k={r.k}: {type(e).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    if not out:
        raise SystemExit(f"failed {r.task} k={r.k}")
    made += 1
    print(f"PREFLIGHT [{r.task}] k={int(r.k):02d}: {r.phrase!r} -> {out!r}", flush=True)
    rows.append({"task": r.task, "k": int(r.k), "arm": "rr16_rulesv3_gemini",
                 "phrase": out, "instruction": r.phrase})

df = pd.DataFrame(rows)
df.to_parquet(OUT, index=False)
print(f"ALL-RULESV3-GEMINI-DONE n={len(df)} (recovered {len(done)}, generated {made}) -> {OUT}")
