#!/usr/bin/env python3
"""A39: gemini traces for the human-naturals phrases (FINAL_EVAL=1 gated).

Standard trace recipe (PREREG A39): CoVer-template prompt via
phrase_rl.cover_prompt, gemini-3.5-flash, temperature 0.4, scene image,
extract_trace; trace generated FROM THE HUMAN PHRASE, never the canonical.
Resume-safe: existing (task, phrase) rows are kept and skipped.
Output: results/human_naturals/a39_traces.parquet (task, register, phrase, trace)
"""
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from google import genai
from google.genai import types

from phrase_rl.cover_prompt import build_user_prompt, extract_trace, load_system_prompt
from phrase_rl.gemini_redteam_assets import call_with_retry  # noqa: E402

R = pathlib.Path(__file__).resolve().parents[1]
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

ph = pd.read_parquet(R / "results/human_naturals/a39_human_phrases.parquet")
frames = pd.read_parquet(R / "results/sealed/sealed_frames.parquet")
img_by_task = {r.task: bytes(r.image_png) for r in frames.itertuples()}

out_p = R / "results/human_naturals/a39_traces.parquet"
done = pd.read_parquet(out_p) if out_p.exists() else pd.DataFrame(
    columns=["task", "register", "phrase", "trace"])
have = set(zip(done.task, done.phrase))
todo = ph[~ph.apply(lambda r: (r.task, r.phrase) in have, axis=1)]
todo = todo.drop_duplicates(subset=["task", "phrase"])
print(f"traces: {len(done)} done, {len(todo)} to generate")
for r in todo.itertuples():
    print(f"  PREFLIGHT {r.task}/{r.register}: {r.phrase!r}")


def one(r):
    img = types.Part.from_bytes(data=img_by_task[r.task], mime_type="image/png")
    raw = call_with_retry(client, types, "gemini-3.5-flash",
                          [img, build_user_prompt(r.phrase, 1)],
                          system=load_system_prompt(), temperature=0.4,
                          max_tokens=2000)
    return dict(task=r.task, register=r.register, phrase=r.phrase,
                trace=extract_trace(raw) or raw[:1200])


rows = list(done.to_dict("records"))
with ThreadPoolExecutor(max_workers=4) as ex:
    for i, rec in enumerate(ex.map(one, todo.itertuples())):
        rows.append(rec)
        if (i + 1) % 25 == 0:
            pd.DataFrame(rows).to_parquet(out_p, index=False)  # checkpoint
            print(f"  {i + 1}/{len(todo)}", flush=True)
pd.DataFrame(rows).to_parquet(out_p, index=False)
print("traces ->", out_p, len(rows))
