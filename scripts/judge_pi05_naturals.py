#!/usr/bin/env python3
"""Semantic meaning judge for the pi0.5/LIBERO sealed NATURAL set.

Port of judge_naturals_v2.py (A33): same prompt, same model, same "judge every
candidate, kept and mechanically-rejected, under one uniform standard" rule.
The mechanical content-word gate rejects legitimate synonyms, so it is the judge
-- not the gate -- that decides admission.

  FINAL_EVAL=1 .venv/bin/python scripts/judge_pi05_naturals.py
  -> results/sealed/ph_pi05_natural_v2_img_judged.parquet
"""
import concurrent.futures as cf
import json
import os
import pathlib

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

REPO = pathlib.Path(__file__).resolve().parents[1]
SEALED = json.load(open(REPO / "results/analysis/pi05_bank/eval_set.json"))["tasks"]
PART = REPO / "results/sealed/pi05_natural_v2_parts_img"
OUT = REPO / "results/sealed/ph_pi05_natural_v2_img_judged.parquet"

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
    cands.append(pd.read_parquet(f).assign(origin="kept"))
for f in sorted(PART.glob("*_rejects.csv")):
    d = pd.read_csv(f)
    cands.append(d[["task", "author", "phrase"]].assign(origin="mech_rejected"))
allc = pd.concat(cands, ignore_index=True).drop_duplicates(["task", "phrase"])
print(f"judging {len(allc)} candidates "
      f"({(allc.origin=='kept').sum()} kept, "
      f"{(allc.origin=='mech_rejected').sum()} mech-rejected)", flush=True)

from google import genai
from google.genai import types
cl = genai.Client()


def _judge_one(r):
    """One verdict. Model/config UNCHANGED from the A33 judge this ports -- the
    instrument must stay identical because the two natural sets get compared."""
    p = PROMPT.format(ref=SEALED[r.task], cand=r.phrase)
    for attempt in range(3):
        try:
            resp = cl.models.generate_content(
                model="gemini-pro-latest", contents=p,
                config=types.GenerateContentConfig(
                    temperature=0.0, http_options=types.HttpOptions(timeout=120_000)))
            return (resp.text or "").strip().upper().startswith("YES")
        except Exception as e:
            print(f"  judge retry {attempt} on {r.phrase!r}: {type(e).__name__}", flush=True)
    return False


# Parallel only -- verdicts are independent and temperature is 0.0, so this
# changes wall-clock, not results. Serial, 351 candidates took ~30-60 min.
# Worker count follows the RULES_APPLY_WORKERS precedent.
JUDGE_WORKERS = int(os.environ.get("JUDGE_WORKERS", "8"))
rows = list(allc.itertuples())
with cf.ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
    verdicts = list(ex.map(_judge_one, rows))
allc["meaning_ok"] = verdicts
allc.to_parquet(OUT, index=False)
kept = allc[allc.meaning_ok]
print(f"judge passed {len(kept)}/{len(allc)} over {kept.task.nunique()} tasks; "
      f"re-admitted {int(((allc.origin=='mech_rejected') & allc.meaning_ok).sum())}; "
      f"rejected {int(((allc.origin=='kept') & ~allc.meaning_ok).sum())} previously kept",
      flush=True)
for r in allc[(allc.origin == 'kept') & (~allc.meaning_ok)].itertuples():
    print(f"  JUDGE-REJECT [{r.task}] {r.phrase}", flush=True)
print(kept.groupby("task").size().to_string(), flush=True)
