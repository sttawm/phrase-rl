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
    Z = np.full_like(G, np.nan)
    ki = {k: i for i, k in enumerate(keys)}
    ei = {e: i for i, e in enumerate(eps)}
    ti = {x: i for i, x in enumerate(ts)}
    for r in sub.itertuples():
        G[ki[r.phrase], ei[r.episode_index], ti[r.t]] = r.grip_row
        Z[ki[r.phrase], ei[r.episode_index], ti[r.t]] = r.z_row
    states[t] = dict(G=G, Z=Z, ki=ki, n_eps=len(eps),
                     t_avail={ei[e]: np.where(~np.isnan(G[:, ei[e], :]).all(axis=0))[0] for e in eps})

# all pairs with their gap, no filtering
allpairs = {t: [] for t in states}
for t, s in states.items():
    ph = [(p, *gt[(t, p)]) for p in s["ki"] if (t, p) in gt]
    for (p1, s1, _), (p2, s2, _) in itertools.combinations(ph, 2):
        better, worse = (p1, p2) if s1 > s2 else (p2, p1)
        allpairs[t].append((better, worse, abs(s1 - s2)))

BLENDS = ("ens100", "z75g25", "z50g50", "c4b", "grip")
rng = np.random.default_rng(11)
hits = {bl: {t: {b[0]: [0, 0] for b in BANDS} for t in states} for bl in BLENDS}
uncond = {bl: [0, 0] for bl in BLENDS}
for _ in range(B):
    for t, s in states.items():
        elig = [e for e in range(s["n_eps"]) if len(s["t_avail"][e]) >= 1]
        epick = rng.choice(elig, size=min(C, len(elig)), replace=False)
        gsel, zsel = [], []
        for e in epick:
            fp = rng.choice(s["t_avail"][e], size=min(F, len(s["t_avail"][e])), replace=False)
            gsel.append(np.nanmean(s["G"][:, e, :][:, fp], axis=1))
            zsel.append(np.nanmean(s["Z"][:, e, :][:, fp], axis=1))
        g = np.nanmean(np.stack(gsel), axis=0)
        z = np.nanmean(np.stack(zsel), axis=0)
        r01 = lambda x: np.argsort(np.argsort(x)) / max(len(x) - 1, 1)
        g01, z01 = r01(-g), r01(z)
        SC = {"ens100": z01, "z75g25": 0.75 * z01 + 0.25 * g01,
              "z50g50": 0.5 * z01 + 0.5 * g01,
              "c4b": 0.25 * z01 + 0.75 * g01, "grip": g01}
        for bl in BLENDS:
            sc = SC[bl]
            for better, worse, gap in allpairs[t]:
                sb, sw = sc[s["ki"][better]], sc[s["ki"][worse]]
                if np.isnan(sb) or np.isnan(sw):
                    continue
                ok = int(sb > sw)
                uncond[bl][0] += ok
                uncond[bl][1] += 1
                for name, lo, hi in BANDS:
                    if lo <= gap < hi:
                        hits[bl][t][name][0] += ok
                        hits[bl][t][name][1] += 1

present = [t for t in states if all(hits["grip"][t][b[0]][1] > 0 for b in BANDS)]
out = {}
for mode in ("raw", "balanced"):
    print(f"\n=== {mode.upper()} — F={F} C={C}, {len(states)} tasks"
          + ("" if mode == "raw" else f", macro-avg over {len(present)} tasks") + " ===")
    print(f"{'gap band':<12} " + "".join(f"{b:>9}" for b in BLENDS) + f"{'pairs':>8}")
    for name, lo, hi in BANDS:
        cells = []
        for bl in BLENDS:
            if mode == "raw":
                hh = sum(hits[bl][t][name][0] for t in states)
                nn = sum(hits[bl][t][name][1] for t in states)
                cells.append(100 * hh / nn if nn else float("nan"))
            else:
                cells.append(float(np.mean([100 * hits[bl][t][name][0] / hits[bl][t][name][1]
                                            for t in present])))
        nn = sum(hits["grip"][t][name][1] for t in states) // B
        out[f"{mode}|{name}"] = {bl: round(c, 1) for bl, c in zip(BLENDS, cells)}
        print(f"{name:<12} " + "".join(f"{c:9.1f}" for c in cells) + f"{nn:8d}")

print("\nUNCONDITIONAL (every pair, no filter): " +
      "  ".join(f"{bl} {100 * uncond[bl][0] / uncond[bl][1]:.1f}" for bl in BLENDS))
out["unconditional"] = {bl: round(100 * uncond[bl][0] / uncond[bl][1], 1) for bl in BLENDS}

print("\nper-task pair counts and grip accuracy (why raw != balanced):")
for t in sorted(states, key=lambda t: -sum(hits["grip"][t][b[0]][1] for b in BANDS)):
    n = sum(hits["grip"][t][b[0]][1] for b in BANDS) // B
    h = sum(hits["grip"][t][b[0]][0] for b in BANDS)
    d = sum(hits["grip"][t][b[0]][1] for b in BANDS)
    print(f"  {t.replace('widowx_', '')[:30]:32s} pairs={n:5d}  grip acc={100 * h / d:5.1f}")
json.dump(out, open("results/analysis/reward_acc_by_gap.json", "w"), indent=1)
print("-> results/analysis/reward_acc_by_gap.json")
