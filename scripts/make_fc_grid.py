#!/usr/bin/env python3
"""Measured (frames x contexts) grid of C4b sign accuracy on the 68 frozen pairs.

For each cell (F, C): bootstrap-subsample F of the banked 4 timesteps and C of
the banked episodes (SAME episodes for both phrases of a pair — CRN, matching
GRPO's shared-context groups), aggregate C4b per phrase, score the pair sign.
Y-metric is actual sign accuracy (floor 50%). Frames axis is capped at the
banked F<=4 today; the t=32 extraction (task #7) extends it.

  python3 scripts/make_fc_grid.py     (local, CPU, ~1-2 min)
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "results/analysis")
import reward_bakeoff_v3 as bk  # noqa: E402

RNG = np.random.default_rng(7)
B = 400
F_GRID = [1, 2, 4]
C_GRID = [1, 2, 4, 8, 16, 20]

feats = bk.load_features()
ens = bk.VerifierEnsemble("results/checkpoints/verifier_reward_ensemble_4f.json")
feats["z_row"] = ens.member_logits(feats).mean(axis=1)
feats["grip_row"] = [float(np.mean(v)) for v in feats.grip_err]
feats["key"] = [bk.norm_key(p) for p in feats.phrase]

# per task: arrays [key, episode, t]
states = {}
for t, sub in feats.groupby("task"):
    keys = sorted(sub.key.unique())
    eps = sorted(sub.episode_index.unique())
    ts = sorted(sub.t.unique())
    Z = np.full((len(keys), len(eps), len(ts)), np.nan)
    G = np.full_like(Z, np.nan)
    ki = {k: i for i, k in enumerate(keys)}
    ei = {e: i for i, e in enumerate(eps)}
    ti = {x: i for i, x in enumerate(ts)}
    for r in sub.itertuples():
        Z[ki[r.key], ei[r.episode_index], ti[r.t]] = r.z_row
        G[ki[r.key], ei[r.episode_index], ti[r.t]] = r.grip_row
    t_avail = {ei[e]: np.where(~np.isnan(Z[0, ei[e], :]))[0] for e in eps}
    states[t] = dict(Z=Z, G=G, ki=ki, n_eps=len(eps), n_t=len(ts), t_avail=t_avail)

pairs = json.load(open("results/analysis/gate_zero_pairs.json"))["pairs"]
covered = [p for p in pairs if p["task"] in states
           and bk.norm_key(p["better"]) in states[p["task"]]["ki"]
           and bk.norm_key(p["worse"]) in states[p["task"]]["ki"]]
print(f"{len(covered)} covered pairs")

r01 = bk.rank01
acc = {r: np.zeros((len(F_GRID), len(C_GRID))) for r in ("ens100", "z75g25", "z50g50", "c4b", "grip")}
for fi, F in enumerate(F_GRID):
    for ci, C in enumerate(C_GRID):
        RW = ("ens100", "z75g25", "z50g50", "c4b", "grip")
        hits_d = {r: 0 for r in RW}
        tot_d = {r: 0 for r in RW}
        for _ in range(B):
            # per task: one shared (episode, frame) draw — CRN across the whole
            # candidate set, mirroring a GRPO group sharing its contexts
            per_task_scores = {}
            for t, s in states.items():
                epick = RNG.choice(s["n_eps"], size=min(C, s["n_eps"]), replace=False)
                zsel, gsel = [], []
                for e in epick:
                    avail = s["t_avail"][e]  # this episode's own frames
                    fp = RNG.choice(avail, size=min(F, len(avail)), replace=False)
                    zsel.append(np.nanmean(s["Z"][:, e, :][:, fp], axis=1))
                    gsel.append(np.nanmean(s["G"][:, e, :][:, fp], axis=1))
                z = np.nanmean(np.stack(zsel), axis=0)
                g = np.nanmean(np.stack(gsel), axis=0)
                z01, g01 = r01(z), r01(-g)
                per_task_scores[t] = {"ens100": z01, "z75g25": 0.75 * z01 + 0.25 * g01,
                                      "z50g50": 0.5 * z01 + 0.5 * g01,
                                      "c4b": 0.25 * z01 + 0.75 * g01, "grip": g01}
            for p in covered:
                s = states[p["task"]]
                for rw in RW:
                    sc = per_task_scores[p["task"]][rw]
                    sb = sc[s["ki"][bk.norm_key(p["better"])]]
                    sw = sc[s["ki"][bk.norm_key(p["worse"])]]
                    if not (np.isnan(sb) or np.isnan(sw)):
                        hits_d[rw] += int(sb > sw)
                        tot_d[rw] += 1
        for rw in RW:
            acc[rw][fi, ci] = 100 * hits_d[rw] / tot_d[rw]
        print(f"F={F} C={C}: " + "  ".join(f"{r} {acc[r][fi, ci]:.1f}" for r in acc))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(30, 4.6))
PANELS = [("ens100", "100% ensemble"), ("z75g25", "75% ens / 25% grip"), ("z50g50", "50/50"),
          ("c4b", "C4b: 25% ens / 75% grip"), ("grip", "100% grip")]
for ax, (rw, name) in zip(axes, PANELS):
    A = acc[rw]
    im = ax.imshow(A, origin="lower", cmap="viridis", vmin=50, vmax=100, aspect="auto")
    for fi, F in enumerate(F_GRID):
        for ci, C in enumerate(C_GRID):
            ax.text(ci, fi, f"{A[fi, ci]:.0f}", ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(C_GRID)))
    ax.set_xticklabels(C_GRID)
    ax.set_yticks(range(len(F_GRID)))
    ax.set_yticklabels(F_GRID)
    ax.set_xlabel("contexts averaged C (per-task cap: spoon 10, carrot/stack 20, egg 60)")
    ax.set_ylabel("frames F per episode (banked 1-4)")
    ax.set_title(f"{name} — sign accuracy % (floor 50)")
    for cost in [4, 8, 16]:
        pts = [(ci, fi) for fi, F in enumerate(F_GRID) for ci, C in enumerate(C_GRID) if F * C == cost]
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "w--", lw=1, alpha=0.6)
    fig.colorbar(im, ax=ax, label="sign accuracy %")
fig.text(0.01, 0.01, "Bootstrap B=400/cell, frames drawn WITHIN each episode (v2 — v1 t-column bug corrected). Banked: 10-60 eps/task, 1-4 frames/ep. "
         "Training operates at C=1; the exam limit (97%) is the C→20 row. Iso-cost dashes: equal F·C compute.",
         fontsize=7, color="#4a5568")
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("results/charts/fc_grid.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
print("chart -> results/charts/fc_grid.png")
json.dump({"F_grid": F_GRID, "C_grid": C_GRID, **{f"acc_{r}": a.tolist() for r, a in acc.items()}, "B": B,
           "pairs": len(covered)},
          open("results/analysis/fc_grid.json", "w"), indent=1)
