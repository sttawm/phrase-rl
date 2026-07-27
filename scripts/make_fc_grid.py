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
C_GRID = [1, 2, 4, 8, 10]

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
    states[t] = dict(Z=Z, G=G, ki=ki, n_eps=len(eps), n_t=len(ts))

pairs = json.load(open("results/analysis/gate_zero_pairs.json"))["pairs"]
covered = [p for p in pairs if p["task"] in states
           and bk.norm_key(p["better"]) in states[p["task"]]["ki"]
           and bk.norm_key(p["worse"]) in states[p["task"]]["ki"]]
print(f"{len(covered)} covered pairs")

r01 = bk.rank01
acc = np.zeros((len(F_GRID), len(C_GRID)))
for fi, F in enumerate(F_GRID):
    for ci, C in enumerate(C_GRID):
        hits = tot = 0
        for _ in range(B):
            # per task: one shared (episode, frame) draw — CRN across the whole
            # candidate set, mirroring a GRPO group sharing its contexts
            per_task_scores = {}
            for t, s in states.items():
                epick = RNG.choice(s["n_eps"], size=min(C, s["n_eps"]), replace=False)
                fpick = RNG.choice(s["n_t"], size=min(F, s["n_t"]), replace=False)
                z = np.nanmean(s["Z"][:, epick][:, :, fpick], axis=(1, 2))
                g = np.nanmean(s["G"][:, epick][:, :, fpick], axis=(1, 2))
                per_task_scores[t] = 0.25 * r01(z) + 0.75 * r01(-g)
            for p in covered:
                s = states[p["task"]]
                sc = per_task_scores[p["task"]]
                sb = sc[s["ki"][bk.norm_key(p["better"])]]
                sw = sc[s["ki"][bk.norm_key(p["worse"])]]
                if not (np.isnan(sb) or np.isnan(sw)):
                    hits += int(sb > sw)
                    tot += 1
        acc[fi, ci] = 100 * hits / tot
        print(f"F={F} C={C}: {acc[fi, ci]:.1f}%")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8.6, 5.2))
im = ax.imshow(acc, origin="lower", cmap="viridis", vmin=50, vmax=100, aspect="auto")
for fi, F in enumerate(F_GRID):
    for ci, C in enumerate(C_GRID):
        ax.text(ci, fi, f"{acc[fi, ci]:.0f}", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
ax.set_xticks(range(len(C_GRID)))
ax.set_xticklabels(C_GRID)
ax.set_yticks(range(len(F_GRID)))
ax.set_yticklabels(F_GRID)
ax.set_xlabel("contexts averaged C (episodes, CRN-shared across the pair)")
ax.set_ylabel("frames F (capped at banked 4 — extraction extends)")
ax.set_title("MEASURED C4b sign accuracy on 68 frozen pairs (%, floor 50)")
# iso-cost guides: cells with equal F*C
for cost in [4, 8, 16]:
    pts = [(ci, fi) for fi, F in enumerate(F_GRID) for ci, C in enumerate(C_GRID) if F * C == cost]
    if len(pts) > 1:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "w--", lw=1, alpha=0.6)
        ax.annotate(f"F·C={cost}", pts[-1], textcoords="offset points", xytext=(10, 4),
                    color="white", fontsize=7.5, alpha=0.85)
fig.colorbar(im, label="sign accuracy %")
fig.text(0.01, 0.01, "Bootstrap B=400/cell over banked exam features (10-20 eps x 4 t x 8 draws per phrase). "
         "Training operates at C=1; the exam limit (97%) is the C→20 row. Iso-cost dashes: equal F·C compute.",
         fontsize=7, color="#4a5568")
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("results/charts/fc_grid.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
print("chart -> results/charts/fc_grid.png")
json.dump({"F_grid": F_GRID, "C_grid": C_GRID, "acc": acc.tolist(), "B": B,
           "pairs": len(covered)},
          open("results/analysis/fc_grid.json", "w"), indent=1)
