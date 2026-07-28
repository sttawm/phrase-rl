#!/usr/bin/env python3
"""Adversarial-input probe files: same 8 contexts/images, but instruction = the
canonical ERT phrase (from ph_val8_reference) and trace = ERT-derived (CoVer
template on (image, ERT) — the deployment-consistent conditioning for hostile
input, mirroring sealed stage_gemini). Outputs contexts_probe8_adv.parquet +
traces_probe8_adv.parquet.
"""
import sys

import pandas as pd

sys.path.insert(0, "src")
from google import genai
from google.genai import types
from phrase_rl.cover_prompt import build_user_prompt, load_system_prompt, extract_trace
from phrase_rl.gemini_redteam_assets import call_with_retry

ctx = pd.read_parquet("results/phrase_artifacts/contexts_probe8.parquet")
ref = pd.read_parquet("results/phrase_artifacts/ph_val8_reference.parquet")
adv = {r.task: r.phrase for r in ref[ref.arm == "val8_adv"].itertuples()}
client = genai.Client()

crows, trows = [], []
for r in ctx.itertuples():
    ert = adv[r.task]
    raw = call_with_retry(client, types, "gemini-3.5-flash",
                          [types.Part.from_bytes(data=bytes(r.image_png), mime_type="image/png"),
                           build_user_prompt(ert, 1)],
                          system=load_system_prompt(), temperature=0.4, max_tokens=2000)
    trace = extract_trace(raw) or raw[:1200]
    row = r._asdict()
    row.pop("Index", None)
    row["instruction"] = ert
    crows.append(row)
    trows.append({"episode_index": r.episode_index, "t": r.t, "trace": trace})
    print(f"{r.task}: ERT trace {len(trace)} chars")
pd.DataFrame(crows).to_parquet("results/phrase_artifacts/contexts_probe8_adv.parquet", index=False)
pd.DataFrame(trows).to_parquet("results/phrase_artifacts/traces_probe8_adv.parquet", index=False)
print("wrote adv probe files (8 contexts, ERT input + ERT traces)")
