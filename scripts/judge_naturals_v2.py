#!/usr/bin/env python3
"""A33 meaning judge: re-admit A33 candidates using a SEMANTIC check.

The mechanical gate in gen_sealed_naturals_v2.py required every content word of
the canonical to survive, which rejected legitimate natural rephrasings
("ramekin" -> "the white dish", "coke can" -> "the red soda can") and did so
~2x more often on the image-conditioned variant, biasing the very comparison the
variants exist to settle. This judge asks whether a candidate commands the SAME
physical outcome -- same object moved, same destination -- allowing any wording.

Judges every candidate (kept and mechanically-rejected) so both variants are
held to one uniform standard. Usage: judge_naturals_v2.py text|image
Writes results/sealed/ph_sealed_natural_v2{,_img}_judged.parquet
"""
import os
import pathlib
import sys

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")
sys.path.insert(0, "scripts")
from build_sealed_assets import SEALED

REPO = pathlib.Path.home() / "dev/robotics/phrase-rl"
variant = sys.argv[1]
SUF = "" if variant == "text" else "_img"
PART = REPO / f"results/sealed/natural_v2_parts{SUF}"

PROMPT = """A robot will execute exactly one of these instructions. Decide whether
the CANDIDATE commands the same physical outcome as the REFERENCE: the same object
is moved, and it ends up at the same destination.

Different words for the same thing are FINE and expected -- "ramekin"/"white bowl",
"coke can"/"red soda can", "sponge"/"green scrubber" all refer to the same object.
Politeness, questions, extra description, and different verbs are FINE.

Answer NO only if the candidate would make the robot move a DIFFERENT object, or
put it in a DIFFERENT place, or if it is too vague to identify either.

REFERENCE: {ref}
CANDIDATE: {cand}

Answer with one word: YES or NO."""

cands = []
for f in sorted(PART.glob("*.parquet")):
    d = pd.read_parquet(f)
    cands.append(d.assign(origin="kept"))
for f in sorted(PART.glob("*_rejects.csv")):
    d = pd.read_csv(f)
    cands.append(d[["task", "author", "phrase"]].assign(origin="mech_rejected"))
allc = pd.concat(cands, ignore_index=True).drop_duplicates(["task", "phrase"])
print(f"[{variant}] judging {len(allc)} candidates "
      f"({(allc.origin=='kept').sum()} kept, {(allc.origin=='mech_rejected').sum()} mech-rejected)",
      flush=True)

from google import genai
from google.genai import types
cl = genai.Client()
verdicts = []
for r in allc.itertuples():
    p = PROMPT.format(ref=SEALED[r.task], cand=r.phrase)
    try:
        resp = cl.models.generate_content(
            model="gemini-pro-latest", contents=p,
            config=types.GenerateContentConfig(
                temperature=0.0, http_options=types.HttpOptions(timeout=120_000)))
        ok = (resp.text or "").strip().upper().startswith("YES")
    except Exception as e:
        print(f"  judge failed on {r.phrase!r}: {type(e).__name__}", flush=True)
        ok = False
    verdicts.append(ok)
allc["meaning_ok"] = verdicts
out = REPO / f"results/sealed/ph_sealed_natural_v2{SUF}_judged.parquet"
allc.to_parquet(out, index=False)
kept = allc[allc.meaning_ok]
print(f"[{variant}] judge passed {len(kept)}/{len(allc)}; "
      f"re-admitted {int(((allc.origin=='mech_rejected') & allc.meaning_ok).sum())} "
      f"mechanically-rejected; rejected "
      f"{int(((allc.origin=='kept') & ~allc.meaning_ok).sum())} previously kept", flush=True)
for r in allc[(allc.origin == 'kept') & (~allc.meaning_ok)].itertuples():
    print(f"  JUDGE-REJECT [{r.task.replace('widowx_','')}] {r.phrase}", flush=True)
