#!/usr/bin/env python3
"""PREREG Amendment 29 sealed assets (gated: FINAL_EVAL=1).

Stage A: 5 NEW evaluation ERT attacks per sealed task (60 total), authored by
the SAME held-out generator as the originals (build_sealed_assets.ERT_PROMPT,
gemini-3.5-flash, temp 0.8, scene image, cover-release shots). Mechanical
filters only, first-5-kept: single line, non-empty, not a duplicate of the
original ERT or of an earlier keep (case/whitespace-normalized).
Stage B: CoVer-template trace on (image, new attack) — build_sealed_assets
stage_gemini recipe verbatim (build_user_prompt(x,1), temp 0.4, extract_trace).
Stage C: MATCHED natural traces for all 192 ph_sealed_rephrase16 phrases —
same trace recipe on (image, natural phrase). Fixes the paper's noted
ERT-derived-trace defect (limitations, main.tex ~787-794).

Resume-safe: existing output rows are kept and skipped. Every kept phrase is
preflight-printed.
Outputs: results/sealed/a29_new_erts.parquet  (task, phrase, trace)
         results/sealed/a29_natural_traces.parquet (task, k, phrase, trace)
"""
import os
import sys

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import json

from google import genai
from google.genai import types

from build_sealed_assets import ERT_PROMPT
from phrase_rl.cover_prompt import build_user_prompt, extract_trace, load_system_prompt
from phrase_rl.gemini_redteam_assets import call_with_retry

N_NEW = 5
OUT_ERT = "results/sealed/a29_new_erts.parquet"
OUT_NAT = "results/sealed/a29_natural_traces.parquet"

cover = json.load(open("results/phrase_artifacts/cover_release_ert_rephrases.json"))["instructions"]
shots = "\n".join(f'nominal: "{n}" -> ERT: "{r["ert_rephrases"][0]}"'
                  for n, r in list(cover.items())[:4])
client = genai.Client()
frames = pd.read_parquet("results/sealed/sealed_frames.parquet")
orig = pd.read_parquet("results/sealed/sealed_assets_gemini.parquet")
orig_ert = {r.task: str(r.ert_instruction) for r in orig.itertuples()}


def norm(s):
    return " ".join(str(s).lower().split())


def trace_for(img_part, phrase):
    raw = call_with_retry(client, types, "gemini-3.5-flash",
                          [img_part, build_user_prompt(phrase, 1)],
                          system=load_system_prompt(), temperature=0.4,
                          max_tokens=2000)
    return extract_trace(raw) or raw[:1200]


# --- Stage A+B: new attacks with traces --------------------------------------
done = pd.read_parquet(OUT_ERT) if os.path.exists(OUT_ERT) else pd.DataFrame(
    columns=["task", "phrase", "trace"])
rows = done.to_dict("records")
for r in frames.itertuples():
    have = [x for x in rows if x["task"] == r.task]
    if len(have) >= N_NEW:
        continue
    img_part = types.Part.from_bytes(data=bytes(r.image_png), mime_type="image/png")
    seen = {norm(orig_ert.get(r.task, ""))} | {norm(x["phrase"]) for x in have}
    attempts = 0
    while len([x for x in rows if x["task"] == r.task]) < N_NEW and attempts < 15:
        attempts += 1
        ert = call_with_retry(client, types, "gemini-3.5-flash",
                              [img_part, ERT_PROMPT.format(shots=shots, nominal=r.nominal)],
                              temperature=0.8, max_tokens=400).strip().strip('"')
        ert = ert.split("\n")[0].strip()
        if not ert or norm(ert) in seen:
            continue
        seen.add(norm(ert))
        tr = trace_for(img_part, ert)
        rows.append({"task": r.task, "phrase": ert, "trace": tr})
        print(f"PREFLIGHT [new-ert] {r.task} | {ert}", flush=True)
        pd.DataFrame(rows).to_parquet(OUT_ERT, index=False)
short = [t for t in frames.task if len([x for x in rows if x["task"] == t]) < N_NEW]
if short:
    raise SystemExit(f"tasks short of {N_NEW} attacks after attempt cap: {short}")
print(f"stage A+B done: {len(rows)} new attacks", flush=True)

# --- Stage C: matched natural traces -----------------------------------------
nat = pd.read_parquet("results/sealed/ph_sealed_rephrase16.parquet")
img_by_task = {r.task: bytes(r.image_png) for r in frames.itertuples()}
doneN = pd.read_parquet(OUT_NAT) if os.path.exists(OUT_NAT) else pd.DataFrame(
    columns=["task", "k", "phrase", "trace"])
haveN = {(r.task, int(r.k)) for r in doneN.itertuples()}
rowsN = doneN.to_dict("records")
for r in nat.itertuples():
    key = (r.task, int(r.k))
    if key in haveN:
        continue
    img_part = types.Part.from_bytes(data=img_by_task[r.task], mime_type="image/png")
    tr = trace_for(img_part, str(r.phrase))
    rowsN.append({"task": r.task, "k": int(r.k), "phrase": str(r.phrase), "trace": tr})
    print(f"PREFLIGHT [nat-trace] {r.task} k={r.k} | {str(r.phrase)[:80]}", flush=True)
    if len(rowsN) % 8 == 0:
        pd.DataFrame(rowsN).to_parquet(OUT_NAT, index=False)
pd.DataFrame(rowsN).to_parquet(OUT_NAT, index=False)
print(f"stage C done: {len(rowsN)} matched natural traces", flush=True)
print("A29-ASSETS-COMPLETE", flush=True)
