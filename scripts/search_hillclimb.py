#!/usr/bin/env python3
"""Round-2+ hill-climb for the reward-guided phrase search (user 2026-07-29:
oracle-style exploration — generate MORE where promising, depth-first).

Consumes funnel rows (results/analysis/search_results.jsonl) AS THEY STREAM,
most-promising-first among the unprocessed (largest round-1 delta). Per
instruction, up to MAX_ROUNDS climbing rounds:
  gen: K image-conditioned variants OF THE CURRENT WINNER (same club frame,
       tagged as the incumbent original wording)
  score: variants + incumbent on the SAME finals episodes the funnel used
         (finals_eps recorded in its row; CRN scorer -> exactly comparable)
  climb: adopt the best variant if it improves grip by > MIN_GAIN, else stop.
Output: results/analysis/search_hillclimb.jsonl (one row per instruction with
the full climb trajectory). Runs on pod6 (score server + .venv-gen + Qwen fits
alongside: ~32G of 49G).
  PYTHONPATH=src .venv-gen/bin/python scripts/search_hillclimb.py <adapter_dir> --ipc-dir /workspace/ipc6
"""
import argparse
import io
import json
import os
import re
import subprocess
import time
import uuid
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import sys
sys.path.insert(0, "src")
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from phrase_rl.cover_prompt import build_single_phrase_prefix
from phrase_rl.phase2_train import apply_template, score_phrases

ap = argparse.ArgumentParser()
ap.add_argument("adapter_dir")
ap.add_argument("--ipc-dir", required=True)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--min-gain", type=float, default=0.002)
ap.add_argument("--out", default="results/analysis/search_hillclimb.jsonl")
args = ap.parse_args()

TAG = "[input: original wording]"
_TAG_RE = re.compile(r"^\s*(?:\[input:[^\]]*\]\s*)+")
sargs = Namespace(k=8, score_seed=0, tau_min=0.0, reward_mode="verifier",
                  k_l2=4, score_timeout=3600, _reward_frames_map=None)
ipc = Path(args.ipc_dir)

proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
base = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False).eval()
ctx = pd.read_parquet("data/contexts_club.parquet")


def gen_variants(seed_phrase, img, k):
    msgs = build_single_phrase_prefix(f"{TAG} {seed_phrase}", img, trace=None)
    inp = apply_template(proc, msgs, continue_final_message=True).to(model.device)
    plen = inp["input_ids"].shape[1]
    with torch.no_grad():
        ss = model.generate(**inp, do_sample=True, temperature=1.0,
                            num_return_sequences=k + 4, max_new_tokens=48)
    out, seen = [], {seed_phrase.strip().lower()}
    for o in ss:
        p = _TAG_RE.sub("", proc.decode(o[plen:], skip_special_tokens=True)
                        .strip().split("\n")[0]).strip()
        if p and len(p.split()) >= 3 and p.lower() not in seen and len(out) < k:
            seen.add(p.lower())
            out.append(p)
    return out


def grips_for(frames, phrases):
    G = []
    for i in range(0, len(frames), 36):
        batch = frames[i:i + 36]
        job = f"hill_{uuid.uuid4().hex[:8]}"
        score_phrases(ipc, job, [(fr, phrases) for fr in batch], sargs)
        G.extend(getattr(score_phrases, "last_grips", None))
    return np.nanmean(np.asarray(G, dtype=float), axis=0)


done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["instruction"] for l in open(args.out)}
    print(f"resume: {len(done)} instructions already climbed")

idle = 0
while True:
    rows = []
    if os.path.exists("results/analysis/search_results.jsonl"):
        rows = [json.loads(l) for l in open("results/analysis/search_results.jsonl")]
    todo = sorted((r for r in rows if r["instruction"] not in done),
                  key=lambda r: -r["delta"])
    if not todo:
        alive = subprocess.run(["tmux", "has-session", "-t", "funnel"],
                               capture_output=True).returncode == 0
        if not alive:
            print("funnel done and all rows climbed — exiting")
            break
        idle += 1
        time.sleep(120)
        continue
    idle = 0
    r = todo[0]
    ins = r["instruction"]
    sub = ctx[ctx.instruction == ins]
    fin_eps = r.get("finals_eps")
    if fin_eps:
        frames = sub[sub.episode_index.isin(fin_eps)].to_dict("records")
    else:  # early funnel rows predate finals_eps recording — draw a fixed set
        eps = sorted(sub.episode_index.unique())
        frames = sub[sub.episode_index.isin(
            list(np.random.default_rng(23).choice(eps, size=min(10, len(eps)), replace=False)))].to_dict("records")
    img = Image.open(io.BytesIO(sub.iloc[0].image_png)).convert("RGB")

    incumbent, inc_grip = r["winner"], r["winner_grip"]
    traj = [{"round": 0, "phrase": incumbent, "grip": inc_grip}]
    for rd in range(1, args.rounds + 1):
        variants = gen_variants(incumbent, img, args.k)
        if not variants:
            break
        g = grips_for(frames, variants + [incumbent])
        gv, gi = g[:-1], g[-1]
        best = int(np.argmin(gv))
        traj.append({"round": rd, "phrase": variants[best], "grip": round(float(gv[best]), 5),
                     "incumbent_rescore": round(float(gi), 5)})
        if gv[best] < gi - args.min_gain:
            incumbent, inc_grip = variants[best], float(gv[best])
        else:
            break
    rec = {"instruction": ins, "orig_grip": r["orig_grip"], "round1_winner": r["winner"],
           "final_winner": incumbent, "final_grip": round(float(inc_grip), 5),
           "total_delta": round(float(r["orig_grip"] - inc_grip), 5),
           "rounds_run": len(traj) - 1, "trajectory": traj}
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    done.add(ins)
    print(f"[{len(done)}] {ins[:46]} rounds={rec['rounds_run']} "
          f"final_delta={rec['total_delta']:.4f}", flush=True)
    if len(done) % 10 == 0:
        os.system(f"timeout 300 bash -c 'git add {args.out} && git commit -q -m \"hillclimb progress [pod]\" && git pull -q --rebase && git push -q' >/dev/null 2>&1")

os.system(f"timeout 300 bash -c 'git add {args.out} && git commit -q -m \"hillclimb complete [pod]\" && git pull -q --rebase && git push -q' >/dev/null 2>&1")
print("HILLCLIMB-COMPLETE")
