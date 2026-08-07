#!/usr/bin/env python3
"""24 natural rephrases per val-8 task — one per layout — for the v11 checkpoint probe.

The original probe used ONE natural input per task, so a checkpoint cell rested
on 8 phrases and only moved when a greedy argmax flipped. Pairing a DIFFERENT
natural input with each of the 24 layouts gives 192 distinct (input, rewrite)
pairs per checkpoint at identical rollout cost.

Same generator and prompt as the sealed natural set (gemini-pro-latest,
image-conditioned, knowledge-free), asked in two batches of 12 with a top-up
pass, deduped case-insensitively, then assigned to layouts in generation order.

  GEMINI_API_KEY=... FINAL_EVAL=1 .venv/bin/python scripts/gen_nat24_probe.py
"""
import os
import time

import pandas as pd
from google import genai
from google.genai import types

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

K = 24
MODEL = "gemini-pro-latest"
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

USER = """Given the original instruction: "{src}", and the attached image, generate {n} reworded instructions that convey the same objective.

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
{src}"""


def ask(client, img_part, src, n):
    resp = client.models.generate_content(
        model=MODEL, contents=[img_part, USER.format(src=src, n=n)],
        config=types.GenerateContentConfig(
            temperature=1.0, system_instruction=SYSTEM,
            http_options=types.HttpOptions(timeout=120_000)))
    out = []
    for line in (resp.text or "").split("\n"):
        line = line.strip().lstrip("0123456789.) ").strip().strip('"')
        if line and not line.lower().startswith("reworded"):
            out.append(line)
    return out


client = genai.Client()
ctx = pd.read_parquet("results/phrase_artifacts/contexts_probe8.parquet")
rows = []
for r in ctx.itertuples():
    part = types.Part.from_bytes(data=r.image_png, mime_type="image/png")
    seen, cands = set(), []
    for attempt in range(6):
        try:
            got = ask(client, part, r.instruction, 12)
        except Exception as e:
            print(f"RETRY {r.task}: {type(e).__name__}", flush=True)
            time.sleep(8 * (attempt + 1))
            continue
        for g in got:
            if g.lower() not in seen:
                seen.add(g.lower())
                cands.append(g)
        if len(cands) >= K:
            break
    if len(cands) < K:
        raise SystemExit(f"{r.task}: only {len(cands)}/{K} unique naturals")
    d = {k: v for k, v in r._asdict().items() if k != "Index"}
    for ep in range(K):
        rec = dict(d)
        rec["instruction"] = cands[ep]
        rec["episode_id"] = ep
        rec["nominal"] = r.instruction
        rows.append(rec)
    print(f"[{r.task}] {len(cands)} unique; layout0={cands[0]!r}  layout23={cands[K - 1]!r}", flush=True)

df = pd.DataFrame(rows)
df.to_parquet("results/phrase_artifacts/contexts_probe8_nat24.parquet", index=False)
print(f"NAT24-PROBE-DONE rows={len(df)} tasks={df.task.nunique()} "
      f"unique_instructions={df.instruction.nunique()}")
