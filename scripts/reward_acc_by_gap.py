#!/usr/bin/env python3
"""Reward ranking accuracy as a function of ground-truth gap — the number you
need to rank an ARBITRARY pair, which neither headline exam number answers
(both condition on the pair being distinguishable).

Two views, because pooling across gap bands is confounded: the ordering-
confidence filter admits different TASKS into different bands, and tasks differ
wildly in how legible their gripper signal is.
  raw     = pooled over all pairs in the band (what the old grids reported)
  balanced= macro-average over tasks present in every band (task mix held fixed)

Also reports the unconditional number: every pair, no gap or confidence filter.

  .venv/bin/python scripts/reward_acc_by_gap.py
"""
import itertools
import json
import math

import numpy as np
import pandas as pd

B = 300
F, C = 4, 20
BANDS = [("0-5 (ties)", 0, 5), ("5-10", 5, 10), ("10-15", 10, 15),
         ("15-25", 15, 25), ("25+", 25, 1e9)]

feats = pd.concat([pd.read_parquet("results/analysis/fine_exam_features_oov.parquet"),
                   pd.read_parquet("results/analysis/fine_exam_features_native.parquet")],
                  ignore_index=True)
pan = pd.read_parquet("results/analysis/fine_exam_phrases.parquet")
gt = {(r.task, r.phrase): (r.gt_success, r.gt_n) for r in pan.itertuples()}

states = {}
for t, sub in feats.groupby("task"):
    keys = sorted(sub.phrase.unique())
    eps = sorted(sub.episode_index.unique())
    ts = sorted(sub.t.unique())
    G = np.full((len(keys), len(eps), len(ts)), np.nan)
    ki = {k: i for i, k in enumerate(keys)}
    ei = {e: i for i, e in enumerate(eps)}
    ti = {x: i for i, x in enumerate(ts)}
    for r in sub.itertuples():
        G[ki[r.phrase], ei[r.episode_index], ti[r.t]] = r.grip_row
    states[t] = dict(G=G, ki=ki, n_eps=len(eps),
                     t_avail={ei[e]: np.where(~np.isnan(G[:, ei[e], :]).all(axis=0))[0] for e in eps})

# all pairs with their gap, no filtering
allpairs = {t: [] for t in states}
for t, s in states.items():
    ph = [(p, *gt[(t, p)]) for p in s["ki"] if (t, p) in gt]
    for (p1, s1, _), (p2, s2, _) in itertools.combinations(ph, 2):
        better, worse = (p1, p2) if s1 > s2 else (p2, p1)
        allpairs[t].append((better, worse, abs(s1 - s2)))

rng = np.random.default_rng(11)
hits = {t: {b[0]: [0, 0] for b in BANDS} for t in states}
uncond = [0, 0]
for _ in range(B):
    for t, s in states.items():
        elig = [e for e in range(s["n_eps"]) if len(s["t_avail"][e]) >= 1]
        epick = rng.choice(elig, size=min(C, len(elig)), replace=False)
        gsel = []
        for e in epick:
            fp = rng.choice(s["t_avail"][e], size=min(F, len(s["t_avail"][e])), replace=False)
            gsel.append(np.nanmean(s["G"][:, e, :][:, fp], axis=1))
        g = np.nanmean(np.stack(gsel), axis=0)
        sc = np.argsort(np.argsort(-g)) / max(len(g) - 1, 1)  # rank01, lower grip err = better
        for better, worse, gap in allpairs[t]:
            sb, sw = sc[s["ki"][better]], sc[s["ki"][worse]]
            if np.isnan(sb) or np.isnan(sw):
                continue
            ok = int(sb > sw)
            uncond[0] += ok
            uncond[1] += 1
            for name, lo, hi in BANDS:
                if lo <= gap < hi:
                    hits[t][name][0] += ok
                    hits[t][name][1] += 1

present = [t for t in states if all(hits[t][b[0]][1] > 0 for b in BANDS)]
print(f"grip-only F={F} C={C}; {len(states)} tasks, {len(present)} have pairs in every band\n")
print(f"{'gap band':<12} {'raw %':>7} {'balanced %':>11} {'pairs':>7}")
rows = {}
for name, lo, hi in BANDS:
    hh = sum(hits[t][name][0] for t in states)
    nn = sum(hits[t][name][1] for t in states)
    per = [100 * hits[t][name][0] / hits[t][name][1] for t in present]
    raw = 100 * hh / nn if nn else float("nan")
    bal = float(np.mean(per)) if per else float("nan")
    rows[name] = dict(raw=round(raw, 1), balanced=round(bal, 1), pairs=nn // B)
    print(f"{name:<12} {raw:7.1f} {bal:11.1f} {nn // B:7d}")

u = 100 * uncond[0] / uncond[1]
print(f"\nUNCONDITIONAL (every pair, no gap/confidence filter): {u:.1f}%  "
      f"over {uncond[1] // B} pairs")
json.dump({"bands": rows, "unconditional": round(u, 1),
           "tasks_all_bands": present}, open("results/analysis/reward_acc_by_gap.json", "w"), indent=1)
print("-> results/analysis/reward_acc_by_gap.json")
