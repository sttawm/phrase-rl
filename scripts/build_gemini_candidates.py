#!/usr/bin/env python3
"""Gemini candidate arm for the training-corpus phrase search (user 2026-07-29:
mirror the val oracle — a vision-grounded author exploring beyond the policy's
distribution, esp. scene-grounded renames).

For each instruction already in search_candidates.parquet: one Gemini call with
a club frame image -> 8 alternative phrasings (rules-informed prompt: concrete
visible colors, common household object names, imperative). Appends rows with
source='gemini' and rewrites the parquet. Run on a pod with GEMINI_API_KEY in
env and data/contexts_club.parquet present (pod5).
"""
import io
import sys

import pandas as pd

sys.path.insert(0, "src")
from google import genai
from google.genai import types

from phrase_rl.gemini_redteam_assets import call_with_retry

PROMPT = """You are helping phrase instructions for a robot arm policy trained on kitchen-manipulation demonstrations (BridgeData). The attached image shows the robot's actual camera view for this task.

Current instruction: "{nominal}"

Write 8 alternative phrasings the policy is likely to follow RELIABLY. Guidelines learned from prior experiments:
- Name objects with common household words the training data would contain; if an object's name is unusual, rename it to what it looks like (e.g. "ramekin" -> "white bowl").
- Add concrete color/appearance adjectives ONLY for objects visible in the image.
- Keep the imperative form and the same task meaning. Keep each under 15 words.
- Vary structure across the 8 (some minimal edits, some full rewrites).

Style examples of phrasings that performed well elsewhere:
{shots}

Output exactly 8 lines, one phrasing per line, no numbering, no quotes."""

client = genai.Client()
cand = pd.read_parquet("results/analysis/search_candidates.parquet")
if (cand.source == "gemini").any():
    print(f"resume: {int((cand.source == 'gemini').sum())} gemini rows already present")
ctx = pd.read_parquet("data/contexts_club.parquet")
ref = pd.read_parquet("results/phrase_artifacts/ph_val8_reference.parquet")
shots = "\n".join(f'- "{p}"' for p in ref[ref.arm == "val8_oracle"].phrase.head(3))

targets = [i for i in cand.instruction.unique()
           if not ((cand.instruction == i) & (cand.source == "gemini")).any()]
print(f"{len(targets)} instructions need gemini candidates")

rows = cand.to_dict("records")
for n, ins in enumerate(targets):
    img = ctx[ctx.instruction == ins].iloc[0].image_png
    part = types.Part.from_bytes(data=bytes(img), mime_type="image/png")
    try:
        out = call_with_retry(client, types, "gemini-3.5-flash",
                              [part, PROMPT.format(nominal=ins, shots=shots)],
                              temperature=0.9, max_tokens=600)
    except Exception as e:
        print(f"  [skip] {ins[:40]}: {type(e).__name__}")
        continue
    seen = {ins.strip().lower()}
    kept = 0
    for line in out.splitlines():
        p = line.strip().strip('"').strip("-• ").strip()
        if p and 3 <= len(p.split()) <= 18 and p.lower() not in seen and kept < 8:
            seen.add(p.lower())
            rows.append({"instruction": ins, "phrase": p, "source": "gemini"})
            kept += 1
    if (n + 1) % 20 == 0:
        print(f"  [{n+1}/{len(targets)}] {len(rows)} total rows", flush=True)
        pd.DataFrame(rows).to_parquet("results/analysis/search_candidates.parquet", index=False)

pd.DataFrame(rows).to_parquet("results/analysis/search_candidates.parquet", index=False)
d = pd.DataFrame(rows)
print(f"-> {len(d)} rows ({(d.source == 'gemini').sum()} gemini)")
