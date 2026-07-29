#!/usr/bin/env python3
"""Transform-probe grid (user 2026-07-29: find insightful phrase gaps NOT
anchored on the ground-truth phrase, for rule derivation).

For each sampled instruction: Gemini applies EXACTLY ONE linguistic transform
per variant (one line each); base + variants scored on the same club contexts
(CRN) -> paired per-transform deltas. Streaming depth-first jsonl; aggregate
later into a rules table with effect sizes.
  PYTHONPATH=src .venv-gen/bin/python scripts/probe_transforms.py --ipc-dir /workspace/ipc6
"""
import argparse
import json
import os
import time
import uuid
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "src")
from google import genai
from google.genai import types as gtypes

from phrase_rl.gemini_redteam_assets import call_with_retry
from phrase_rl.phase2_train import score_phrases

TRANSFORMS = [
    ("color_adj", "Add a correct color/appearance adjective (from the image) to each object mentioned; change nothing else."),
    ("rename_visual", "Rename each object to what it visually looks like in the image using a common household word; change nothing else."),
    ("verb_synonym", "Replace the main verb with a natural synonym; change nothing else."),
    ("drop_articles", "Remove all articles (a/an/the); change nothing else."),
    ("add_spatial", "Add one short, correct spatial detail visible in the image (e.g. 'on the left'); change nothing else."),
    ("minimalize", "Shorten to the fewest words that keep the same meaning."),
    ("elaborate", "Expand into a longer, more formal sentence with the same meaning."),
]

ap = argparse.ArgumentParser()
ap.add_argument("--ipc-dir", required=True)
ap.add_argument("--out", default="results/analysis/transform_probe.jsonl")
ap.add_argument("--n-instructions", type=int, default=50)
ap.add_argument("--screen-f", type=int, default=2)
ap.add_argument("--screen-c", type=int, default=6)
args = ap.parse_args()

sargs = Namespace(k=8, score_seed=0, tau_min=0.0, reward_mode="verifier",
                  k_l2=4, score_timeout=3600, _reward_frames_map=None)
ipc = Path(args.ipc_dir)
if not os.environ.get("GEMINI_API_KEY"):
    for _line in open(os.path.expanduser("~/.bashrc")):
        if _line.startswith("export GEMINI_API_KEY="):
            os.environ["GEMINI_API_KEY"] = _line.split("=", 1)[1].strip().strip('"').strip("'")
            break
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
RNG = np.random.default_rng(53)

ctx = pd.read_parquet("data/contexts_club.parquet")
order = list(ctx.instruction.unique())
RNG.shuffle(order)
order = order[:args.n_instructions]

PROMPT = """The attached image is a robot arm's camera view. Below is an instruction and a list of {n} transformations. Apply EACH transformation SEPARATELY to the instruction (each applied to the ORIGINAL, not cumulatively).

Instruction: "{ins}"

Transformations:
{tlist}

Output exactly {n} lines: line i = the instruction with ONLY transformation i applied. No numbering, no quotes."""


def grips_for(frames, phrases):
    G = []
    for i in range(0, len(frames), 36):
        batch = frames[i:i + 36]
        job = f"tp_{uuid.uuid4().hex[:8]}"
        score_phrases(ipc, job, [(fr, phrases) for fr in batch], sargs)
        G.extend(getattr(score_phrases, "last_grips", None))
    return np.nanmean(np.asarray(G, dtype=float), axis=0)


done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["instruction"] for l in open(args.out) if l.strip()}
    print(f"resume: {len(done)}")

for ins in order:
    if ins in done:
        continue
    t0 = time.time()
    sub = ctx[ctx.instruction == ins]
    eps = sorted(sub.episode_index.unique())
    sc_eps = list(RNG.choice(eps, size=min(args.screen_c, len(eps)), replace=False))
    frames = sub[sub.episode_index.isin(sc_eps)].groupby("episode_index").head(args.screen_f).to_dict("records")
    img = gtypes.Part.from_bytes(data=bytes(sub.iloc[0].image_png), mime_type="image/png")
    tlist = "\n".join(f"{i+1}. {d}" for i, (_, d) in enumerate(TRANSFORMS))
    try:
        out = call_with_retry(client, gtypes, "gemini-3.5-flash",
                              [img, PROMPT.format(ins=ins, n=len(TRANSFORMS), tlist=tlist)],
                              temperature=0.4, max_tokens=700)
    except Exception as e:
        print(f"  gen error {ins[:30]!r}: {type(e).__name__}")
        continue
    lines = [l.strip().strip('"').strip() for l in out.splitlines() if l.strip()]
    if len(lines) < len(TRANSFORMS):
        print(f"  short output ({len(lines)}) for {ins[:36]!r}, skipping")
        continue
    variants = lines[:len(TRANSFORMS)]
    g = grips_for(frames, [ins] + variants)
    base = float(g[0])
    rec = {"instruction": ins, "base_grip": round(base, 5), "n_ctx_evals": len(frames),
           "transforms": [{"name": TRANSFORMS[i][0], "phrase": variants[i],
                           "grip": round(float(g[i + 1]), 5),
                           "delta": round(base - float(g[i + 1]), 5),
                           "unchanged": variants[i].strip().lower() == ins.strip().lower()}
                          for i in range(len(TRANSFORMS))],
           "sec": round(time.time() - t0, 1)}
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    done.add(ins)
    print(f"[{len(done)}/{len(order)}] {ins[:44]!r} {rec['sec']:.0f}s", flush=True)
    if len(done) % 10 == 0:
        os.system(f"timeout 300 bash -c 'git add {args.out} && git commit -q -m \"transform probe progress [pod]\" && git -c rebase.autoStash=true pull -q --rebase && git push -q' >/dev/null 2>&1")

os.system(f"timeout 300 bash -c 'git add {args.out} && git commit -q -m \"transform probe complete [pod]\" && git -c rebase.autoStash=true pull -q --rebase && git push -q' >/dev/null 2>&1")
print("TRANSFORM-PROBE-COMPLETE")
