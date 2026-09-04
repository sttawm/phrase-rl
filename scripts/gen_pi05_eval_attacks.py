#!/usr/bin/env python3
"""pi0.5 sealed ADVERSARIAL set (ERT). Gated: FINAL_EVAL=1.

Faithful to PHRASE-GENERATION-RECIPE.md section 2: ERT_PROMPT from
build_sealed_assets.py verbatim, gemini-3.5-flash, temperature 0.8,
max_tokens 400, IMAGE ATTACHED, few-shots = the first four nominal->ERT pairs
from the CoVer release. Held-out by construction: the training attacks came from
generate.md text-only, a different generator.

  FINAL_EVAL=1 GEMINI_API_KEY=... .venv/bin/python scripts/gen_pi05_eval_attacks.py --n 3
  -> results/sealed/ph_pi05_adversarial.parquet   (task, phrase, k)
"""
import argparse, json, os, pathlib, sys, time
import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required (sealed generation)")
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_sealed_assets import ERT_PROMPT          # verbatim, do not re-word

D = REPO / "results/analysis/pi05_bank"
OUT = REPO / "results/sealed/ph_pi05_adversarial.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3)
a = ap.parse_args()

ev = json.load(open(D / "eval_set.json"))["tasks"]
frames = {}
for sub in ("eval_frames", "test_frames", "frames"):
    for p in (D / sub).glob("*.png"):
        suite, tid = p.stem.rsplit("__", 1)
        frames.setdefault(f"{suite}:{int(tid)}", p.read_bytes())
missing = [k for k in ev if k not in frames]
if missing:
    raise SystemExit(f"missing frames for {len(missing)} tasks: {missing[:4]}")

cover = json.load(open(REPO / "results/phrase_artifacts/cover_release_ert_rephrases.json"))["instructions"]
shots = "\n".join(f'nominal: "{n}" -> ERT: "{r["ert_rephrases"][0]}"'
                  for n, r in list(cover.items())[:4])

from google import genai
from google.genai import types
cl = genai.Client()
rows = []
for task, canon in ev.items():
    got = []
    for k in range(a.n):
        prompt = ERT_PROMPT.format(shots=shots, nominal=canon)
        if got:   # anti-repetition, mirroring the naturals channel
            prompt += ("\n\nALREADY WRITTEN for this task - do NOT repeat these or "
                       "produce close variants:\n" + "\n".join(f"  {g}" for g in got))
        for attempt in range(5):
            try:
                r = cl.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=[types.Part.from_bytes(data=frames[task], mime_type="image/png"), prompt],
                    # 400 was the bridge budget for a non-thinking model; on
                    # gemini-3.5-flash the thinking tokens are charged to the
                    # same budget and the instruction came back cut mid-clause.
                    # The cap is a truncation guard, not a sampling parameter.
                    config=types.GenerateContentConfig(temperature=0.8, max_output_tokens=2000))
                if getattr(r.candidates[0], "finish_reason", None) is not None \
                        and "MAX_TOKENS" in str(r.candidates[0].finish_reason):
                    print("   truncated (MAX_TOKENS), retrying", flush=True)
                    continue
                # the model wraps its one-sentence answer across lines; taking
                # the first line truncated attacks mid-clause ("...inside the").
                # The prompt asks for exactly one instruction, so collapse the
                # whole reply to a single whitespace-normalised line.
                cand = " ".join((r.text or "").split()).strip().strip('"')
                if cand and cand.lower() not in {g.lower() for g in got}:
                    got.append(cand); break
            except Exception as e:
                print(f"   retry {attempt+1}: {type(e).__name__}", flush=True); time.sleep(3 * (attempt + 1))
    for k, g in enumerate(got):
        rows.append({"task": task, "phrase": g, "k": k})
        print(f"PREFLIGHT [ert] {task} | {g}", flush=True)
df = pd.DataFrame(rows)
df.to_parquet(OUT, index=False)
print(f"\nattacks: {len(df)} over {df.task.nunique()} tasks -> {OUT.relative_to(REPO)}")
