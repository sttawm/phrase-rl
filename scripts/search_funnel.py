#!/usr/bin/env python3
"""Depth-first reward-guided phrase search (user 2026-07-29: complete one
instruction at a time so early termination still yields finished results).

Per instruction, in descending club-size order:
  screen: all candidates at F=1 x C=4   (~65% cell, cheap cut)
  finals: top-3 survivors + original at F=4 x C=10 (~86-92% band)
  emit:   one completed jsonl row (ranked phrases, winner, delta vs original)
Resume: instructions already in the output jsonl are skipped.
Run on a pod with the score server up:
  PYTHONPATH=src .venv-gen/bin/python scripts/search_funnel.py --ipc-dir /workspace/ipc6
"""
import argparse
import json
import os
import uuid
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from phrase_rl.phase2_train import score_phrases

ap = argparse.ArgumentParser()
ap.add_argument("--ipc-dir", required=True)
ap.add_argument("--out", default="results/analysis/search_results.jsonl")
ap.add_argument("--screen-c", type=int, default=4)
ap.add_argument("--finals-c", type=int, default=10)
ap.add_argument("--finals-f", type=int, default=4)
args = ap.parse_args()

sargs = Namespace(k=8, score_seed=0, tau_min=0.0, reward_mode="verifier",
                  k_l2=4, score_timeout=3600, _reward_frames_map=None)
ipc = Path(args.ipc_dir)
RNG = np.random.default_rng(23)

cand = pd.read_parquet("results/analysis/search_candidates.parquet")
ctx = pd.read_parquet("data/contexts_club.parquet")
sizes = ctx.groupby("instruction").episode_index.nunique()
order = [i for i in sizes.sort_values(ascending=False).index if i in set(cand.instruction)]

done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["instruction"] for l in open(args.out)}
    print(f"resume: {len(done)} instructions already complete")


def grips_for(frames, phrases):
    """Score phrases on the given frame-contexts; return mean grip per phrase."""
    G = []
    for i in range(0, len(frames), 36):
        batch = frames[i:i + 36]
        job = f"srch_{uuid.uuid4().hex[:8]}"
        score_phrases(ipc, job, [(fr, phrases) for fr in batch], sargs)
        g = getattr(score_phrases, "last_grips", None)
        G.extend(g)
    return np.nanmean(np.asarray(G, dtype=float), axis=0)


n_done = 0
for ins in order:
    if ins in done:
        continue
    sub = ctx[ctx.instruction == ins]
    eps = sorted(sub.episode_index.unique())
    phrases = list(cand[cand.instruction == ins].phrase)
    src = dict(zip(cand[cand.instruction == ins].phrase, cand[cand.instruction == ins].source))

    sc_eps = list(RNG.choice(eps, size=min(args.screen_c, len(eps)), replace=False))
    sc_frames = [r for r in sub[sub.episode_index.isin(sc_eps)].to_dict("records")][::4][:args.screen_c]
    g_screen = grips_for(sc_frames, phrases)
    keep_idx = np.argsort(g_screen)[:3]
    finalists = sorted(set([phrases[i] for i in keep_idx] + [ins]))

    fin_eps = list(RNG.choice(eps, size=min(args.finals_c, len(eps)), replace=False))
    fin_frames = sub[sub.episode_index.isin(fin_eps)].to_dict("records")
    g_fin = grips_for(fin_frames, finalists)

    ranked = sorted(zip(finalists, g_fin), key=lambda x: x[1])
    orig_g = dict(ranked).get(ins, float("nan"))
    winner, wg = ranked[0]
    rec = {"instruction": ins, "club_eps": len(eps),
           "winner": winner, "winner_grip": round(float(wg), 5),
           "orig_grip": round(float(orig_g), 5),
           "delta": round(float(orig_g - wg), 5),
           "winner_source": src.get(winner, "original"),
           "finals_eps": [int(e) for e in fin_eps],
           "ranked": [{"phrase": p, "grip": round(float(g), 5), "source": src.get(p, "original")}
                      for p, g in ranked]}
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    n_done += 1
    print(f"[{n_done}] {ins[:50]} -> winner={'ORIGINAL' if winner == ins else 'sample'} "
          f"delta={rec['delta']:.4f}", flush=True)
    if n_done % 10 == 0:
        os.system(f"timeout 300 bash -c 'git add {args.out} && git commit -q -m \"search funnel progress [pod]\" && git pull -q --rebase && git push -q' >/dev/null 2>&1")

os.system(f"timeout 300 bash -c 'git add {args.out} && git commit -q -m \"search funnel complete [pod]\" && git pull -q --rebase && git push -q' >/dev/null 2>&1")
print(f"DONE: {n_done} instructions -> {args.out}")
