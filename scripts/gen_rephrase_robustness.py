#!/usr/bin/env python3
"""Rephrase-robustness generation (PREREG Amendment 19).

K=16 natural human rephrases per sealed nominal: 2 sampled calls x 8 at temp 1.0
(gemini-pro-latest, provenance-matched to rows 12/17 + arm E), sealed frame
attached, NO trace. Prompt = CoVer architecture with ALL rules-like content
removed (no simplicity ask, no adverb/color/diversity rules, no few-shots);
register target = natural human phrasing. Dedupe (casefold) to 16; up to 2
top-up calls if short. Preflight-prints every line (prereg requirement).

Writes results/sealed/ph_sealed_rephrase16.parquet (task, arm, phrase, k,
call_id) and results/sealed/rephrase16_meta.json (model, temp, per-call yield).
FINAL_EVAL=1 required.
"""
import json
import os
import re
import time

import pandas as pd
from google import genai
from google.genai import types

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

MODEL = "gemini-pro-latest"
TEMP = 1.0
K = 16
PER_CALL = 8

SYSTEM = """You are a text-transformation assistant for robot manipulation tasks.

You will be given:
- An image of the scene.
- An instruction describing a manipulation goal.

Your task is to:
1. Understand the meaning of the original instruction.
2. Reword the instruction into multiple alternatives that preserve the original intent.

Guidelines:
- Every reworded instruction must be something a person might actually say when asking for this task to be done.
- All reworded instructions must mean the same thing as the original."""

USER = """Given the original instruction: "{src}", and the attached image, generate 8 reworded instructions that convey the same objective.

Guidelines for rephrasing:
1. Write the way real people actually talk when asking for a task — natural, everyday phrasing.
2. Different people phrase the same request differently: vary the phrasing the way different natural speakers would.
3. Each rephrase must keep the same objects and the same goal as the original.
4. You may refer to objects the way they appear in the image, but do not contradict the original instruction or mention things you cannot see.

Format your response as (no analysis, no commentary):
Reworded Instructions:
1. <Alternative phrasing 1>
...

Original Instruction:
{src}

Reworded Instructions:
1."""

client = genai.Client()
frames = pd.read_parquet("results/sealed/sealed_frames.parquet")
rows, meta = [], {"model": MODEL, "temperature": TEMP, "k": K, "calls": []}
for r in frames.itertuples():
    got, call_id = [], 0
    while len(got) < K and call_id < 4:  # 2 planned + up to 2 top-ups
        text = None
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=MODEL,
                    contents=[types.Part.from_bytes(data=bytes(r.image_png), mime_type="image/png"),
                              USER.format(src=r.nominal)],
                    config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=TEMP),
                )
                text = resp.text
                break
            except Exception as e:
                print(f"RETRY {r.task} call{call_id}: {type(e).__name__}", flush=True)
                time.sleep(10 * (attempt + 1))
        if text is None:
            raise SystemExit(f"API failed x3 for {r.task}")
        lines = re.findall(r"^\s*\d+\.\s*(.+?)\s*$", text, re.M)[:PER_CALL]
        fresh = 0
        for l in lines:
            p = l.strip().strip('"').strip()
            if p and p.casefold() not in {g[0].casefold() for g in got}:
                got.append((p, call_id))
                fresh += 1
        meta["calls"].append({"task": r.task, "call": call_id, "parsed": len(lines), "fresh": fresh})
        call_id += 1
    got = got[:K]
    if len(got) < K:
        print(f"SHORT {r.task}: only {len(got)}/{K} after {call_id} calls", flush=True)
    for i, (p, c) in enumerate(got):
        rows.append({"task": r.task, "arm": "rephrase16", "phrase": p, "k": i, "call_id": c})
        print(f"PREFLIGHT [{r.task}] k={i:02d} call={c}: {p!r}", flush=True)

pd.DataFrame(rows).to_parquet("results/sealed/ph_sealed_rephrase16.parquet", index=False)
json.dump(meta, open("results/sealed/rephrase16_meta.json", "w"), indent=1)
print(f"ALL-REPHRASE16-DONE n={len(rows)}")
