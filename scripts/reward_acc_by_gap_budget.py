#!/usr/bin/env python3
"""Ranking accuracy vs ground-truth gap, swept over sampling budget.

F is held at 4 (the production setting) and C swept over {1,2,4,8,16}, giving
total query budgets F x C of 4, 8, 16, 32, 64. Reports both weightings (raw =
pool all pairs; balanced = per-task then average tasks equally), for all five
reward designs.

Pair scoring is vectorized per task (index arrays + per-band masks) so the whole
sweep runs in ~1 minute.

  .venv/bin/python scripts/reward_acc_by_gap_budget.py
"""
import itertools
import json

import numpy as np
import pandas as pd

B = 300
F = 4
C_SWEEP = [1, 2, 4, 8, 16]
BANDS = [("0-5 (ties)", 0, 5), ("5-10", 5, 10), ("10-15", 10, 15),
         ("15-25", 15, 25), ("25+", 25, 1e9)]
BLENDS = ("ens100", "z75g25", "z50g50", "c4b", "grip")

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
    t_avail = {ei[e]: np.where(~np.isnan(G[:, ei[e], :]).all(axis=0))[0] for e in eps}
    # vectorized pair index arrays + band masks
    ph = [(p, *gt[(t, p)]) for p in keys if (t, p) in gt]
    bi, wi, gaps = [], [], []
    for (p1, s1, _), (p2, s2, _) in itertools.combinations(ph, 2):
        b, w = (p1, p2) if s1 > s2 else (p2, p1)
        bi.append(ki[b]); wi.append(ki[w]); gaps.append(abs(s1 - s2))
    gaps = np.array(gaps)
    states[t] = dict(G=G, Z=Z, n_eps=len(eps), t_avail=t_avail,
                     bi=np.array(bi, int), wi=np.array(wi, int),
                     masks={n: (gaps >= lo) & (gaps < hi) for n, lo, hi in BANDS})

rank01 = lambda x: np.argsort(np.argsort(x)) / max(len(x) - 1, 1)
out, printable = {}, []
for C in C_SWEEP:
    rng = np.random.default_rng(11)
    hits = {bl: {t: {n: 0 for n, _, _ in BANDS} for t in states} for bl in BLENDS}
    tot = {t: {n: 0 for n, _, _ in BANDS} for t in states}
    for _ in range(B):
        for t, s in states.items():
            elig = [e for e in range(s["n_eps"]) if len(s["t_avail"][e]) >= 1]
            epick = rng.choice(elig, size=min(C, len(elig)), replace=False)
            gsel, zsel = [], []
            for e in epick:
                av = s["t_avail"][e]
                fp = rng.choice(av, size=min(F, len(av)), replace=False)
                gsel.append(np.nanmean(s["G"][:, e, :][:, fp], axis=1))
                zsel.append(np.nanmean(s["Z"][:, e, :][:, fp], axis=1))
            g01 = rank01(-np.nanmean(np.stack(gsel), axis=0))
            z01 = rank01(np.nanmean(np.stack(zsel), axis=0))
            SC = {"ens100": z01, "z75g25": 0.75 * z01 + 0.25 * g01,
                  "z50g50": 0.5 * z01 + 0.5 * g01,
                  "c4b": 0.25 * z01 + 0.75 * g01, "grip": g01}
            for bl in BLENDS:
                sc = SC[bl]
                wins = sc[s["bi"]] > sc[s["wi"]]
                for n, _, _ in BANDS:
                    hits[bl][t][n] += int(wins[s["masks"][n]].sum())
            for n, _, _ in BANDS:
                tot[t][n] += int(s["masks"][n].sum())
    for mode in ("raw", "balanced"):
        for n, _, _ in BANDS:
            for bl in BLENDS:
                if mode == "raw":
                    v = 100 * sum(hits[bl][t][n] for t in states) / sum(tot[t][n] for t in states)
                else:
                    v = float(np.mean([100 * hits[bl][t][n] / tot[t][n]
                                       for t in states if tot[t][n] > 0]))
                out.setdefault(f"FxC={F * C}", {}).setdefault(mode, {}).setdefault(n, {})[bl] = round(v, 1)
    printable.append((F * C, C))
    row = out[f"FxC={F * C}"]["balanced"]
    print(f"FxC={F * C:>3} (F={F} C={C:>2})  balanced: " +
          "  ".join(f"{bl} {row['25+'][bl]:.0f}/{row['10-15'][bl]:.0f}" for bl in BLENDS)
          + "   [25+ / 10-15pp]")

json.dump(out, open("results/analysis/reward_acc_by_gap_budget.json", "w"), indent=1)
print("-> results/analysis/reward_acc_by_gap_budget.json")
