#!/usr/bin/env python3
"""Fine-discrimination grid: reward sign accuracy on close pairs, by (F, C) and
blend, from fine_exam_features_{half}.parquet. Pairs built from the panel's
rollout ground truth with binomial ordering-confidence; headline = pairs with
confidence >= 0.8, calibration bucket (0.5-5pp) reported raw (should be ~50%).

  python3 scripts/make_fine_grid.py oov      (or: native, all)
"""
import itertools
import json
import math
import sys

import numpy as np
import pandas as pd

HALF = sys.argv[1] if len(sys.argv) > 1 else "oov"
files = {"oov": ["results/analysis/fine_exam_features_oov.parquet"],
         "native": ["results/analysis/fine_exam_features_native.parquet"],
         "all": ["results/analysis/fine_exam_features_oov.parquet",
                 "results/analysis/fine_exam_features_native.parquet"]}[HALF]
feats = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
pan = pd.read_parquet("results/analysis/fine_exam_phrases.parquet")
gt = {(r.task, r.phrase): (r.gt_success, r.gt_n) for r in pan.itertuples()}

RNG = np.random.default_rng(11)
B = 300
F_GRID = [1, 3, 5]
C_GRID = [1, 2, 4, 8, 10, 16]
BLENDS = ("ens100", "z75g25", "z50g50", "c4b", "grip")

states = {}
for t, sub in feats.groupby("task"):
    keys = sorted(sub.phrase.unique())
    eps = sorted(sub.episode_index.unique())
    ts = sorted(sub.t.unique())
    Z = np.full((len(keys), len(eps), len(ts)), np.nan)
    G = np.full_like(Z, np.nan)
    ki = {k: i for i, k in enumerate(keys)}
    ei = {e: i for i, e in enumerate(eps)}
    ti = {x: i for i, x in enumerate(ts)}
    for r in sub.itertuples():
        Z[ki[r.phrase], ei[r.episode_index], ti[r.t]] = r.z_row
        G[ki[r.phrase], ei[r.episode_index], ti[r.t]] = r.grip_row
    t_avail = {ei[e]: np.where(~np.isnan(Z[:, ei[e], :]).all(axis=0))[0] for e in eps}
    states[t] = dict(Z=Z, G=G, ki=ki, n_eps=len(eps), t_avail=t_avail)

# pairs with ordering confidence, bucketed by gap
buckets = {"calib_0.5-5": [], "fine_5-10": [], "med_10-15": []}
for t, s in states.items():
    ph = [(p, *gt[(t, p)]) for p in s["ki"] if (t, p) in gt]
    for (p1, s1, n1), (p2, s2, n2) in itertools.combinations(ph, 2):
        gap = abs(s1 - s2)
        se = 100 * math.sqrt(s1/100*(1-s1/100)/n1 + s2/100*(1-s2/100)/n2)
        conf = 0.5 * (1 + math.erf((gap / se) / math.sqrt(2))) if se > 0 else 1.0
        rec = (t, (p1 if s1 > s2 else p2), (p2 if s1 > s2 else p1), gap, conf)
        if 0.5 <= gap < 5: buckets["calib_0.5-5"].append(rec)
        elif 5 <= gap < 10 and conf >= 0.8: buckets["fine_5-10"].append(rec)
        elif 10 <= gap < 15 and conf >= 0.8: buckets["med_10-15"].append(rec)
for b, prs in buckets.items():
    print(f"{b}: {len(prs)} pairs")

r01 = lambda x: (np.argsort(np.argsort(x)) / max(len(x) - 1, 1))
out = {}
for F in F_GRID:
    # an episode only counts for this F if it actually has F scored frames —
    # no silent F-degradation; tasks with zero eligible eps drop to NaN (pairs skipped)
    elig = {t: [e for e in range(s["n_eps"]) if len(s["t_avail"][e]) >= F]
            for t, s in states.items()}
    print(f"F={F} eligible eps: " + ", ".join(
        f"{t.replace('widowx_', '').replace('_clean', '')}={len(v)}" for t, v in elig.items()))
    for C in C_GRID:
        acc = {bl: {b: [0, 0] for b in buckets} for bl in BLENDS}
        for _ in range(B):
            sc = {}
            for t, s in states.items():
                pool = elig[t]
                if not pool:
                    nanv = np.full(len(s["ki"]), np.nan)
                    sc[t] = {bl: nanv for bl in BLENDS}
                    continue
                epick = RNG.choice(pool, size=min(C, len(pool)), replace=False)
                zs, gs = [], []
                for e in epick:
                    fp = RNG.choice(s["t_avail"][e], size=F, replace=False)
                    zs.append(np.nanmean(s["Z"][:, e, :][:, fp], axis=1))
                    gs.append(np.nanmean(s["G"][:, e, :][:, fp], axis=1))
                z = np.nanmean(np.stack(zs), axis=0); g = np.nanmean(np.stack(gs), axis=0)
                z01, g01 = r01(z), r01(-g)
                sc[t] = {"ens100": z01, "z75g25": 0.75*z01 + 0.25*g01, "z50g50": 0.5*z01 + 0.5*g01, "c4b": 0.25*z01 + 0.75*g01, "grip": g01}
            for b, prs in buckets.items():
                for t, bett, wors, gap, conf in prs:
                    s = states[t]
                    for bl in BLENDS:
                        v = sc[t][bl]
                        sb, sw = v[s["ki"][bett]], v[s["ki"][wors]]
                        if not (np.isnan(sb) or np.isnan(sw)):
                            acc[bl][b][0] += int(sb > sw); acc[bl][b][1] += 1
        out[f"F{F}_C{C}"] = {bl: {b: round(100*h/max(n,1), 1) for b, (h, n) in d.items()}
                            for bl, d in acc.items()}
        line = f"F={F} C={C}: "
        for bl in BLENDS:
            line += f"{bl}[fine {out[f'F{F}_C{C}'][bl]['fine_5-10']} med {out[f'F{F}_C{C}'][bl]['med_10-15']} calib {out[f'F{F}_C{C}'][bl]['calib_0.5-5']}]  "
        print(line)
json.dump({"half": HALF, "B": B, "buckets": {b: len(p) for b, p in buckets.items()},
           "grid": out}, open(f"results/analysis/fine_grid_{HALF}.json", "w"), indent=1)
print(f"-> results/analysis/fine_grid_{HALF}.json")
