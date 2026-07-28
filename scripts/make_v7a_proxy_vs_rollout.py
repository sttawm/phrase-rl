#!/usr/bin/env python3
"""v7a: proxy reward (C4b val) vs REAL rollout success, paired.

Left panel: both curves over training steps (twin axes). Right panel: scatter of
proxy vs rollout at matched steps with phase-split correlations — the user's
question is whether rollouts keep improving after the proxy plateaus.
Proxy: mean_greedy_reward on the 40-context val set (results/analysis/v7a_train_log.jsonl).
Rollouts: 4-task probe curve (results/analysis/v7_curve_backfill.jsonl, n=192) and
FULL val-8 checkpoints (results/analysis/v7_dev8_backfill.jsonl, n=192).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

vals = {r["step"]: r["mean_greedy_reward"]
        for r in (json.loads(l) for l in open("results/analysis/v7a_train_log.jsonl"))
        if r.get("type") == "val"}
c4t = {r["step"]: r["pooled"] for r in (json.loads(l) for l in open("results/analysis/v7_curve_backfill.jsonl"))}
d8 = sorted((json.loads(l) for l in open("results/analysis/v7_dev8_backfill.jsonl")), key=lambda r: r["step"])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5))

s = sorted(vals)
a1.plot(s, [vals[x] for x in s], marker="o", ms=4, color="#805ad5", lw=1.6, label="proxy reward (C4b val, greedy rewrite)")
a1.set_xlabel("v7a step")
a1.set_ylabel("proxy reward (C4b, 40 val contexts)", color="#805ad5")
a1.tick_params(axis="y", labelcolor="#805ad5")
ar = a1.twinx()
cs = sorted(c4t)
ar.plot(cs, [c4t[x] for x in cs], marker=".", ms=6, color="#90cdf4", lw=1.3, label="REAL rollout: 4-task probe (n=192)")
ar.plot([r["step"] for r in d8], [r["pooled"] for r in d8], marker="D", ms=9, ls="-",
        color="#2b6cb0", lw=2.0, label="REAL rollout: FULL val-8 (n=192)")
ar.set_ylabel("rollout success %", color="#2b6cb0")
ar.tick_params(axis="y", labelcolor="#2b6cb0")
h1, l1 = a1.get_legend_handles_labels()
h2, l2 = ar.get_legend_handles_labels()
ar.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
a1.axvspan(0, 100, alpha=0.06, color="#48bb78")
a1.text(8, a1.get_ylim()[1] - 0.04, "learning phase", fontsize=8, color="#2f855a", va="top")
a1.axvspan(100, 340, alpha=0.05, color="#a0aec0")
a1.text(215, a1.get_ylim()[1] - 0.04, "proxy plateau/dip", fontsize=8, color="#718096", va="top")
a1.set_title("both signals over training")
a1.grid(alpha=0.2)

# matched-step scatter, phase-split correlation
ms_ = sorted(set(vals) & set(c4t))
px = np.array([vals[x] for x in ms_]); ry = np.array([c4t[x] for x in ms_])
early = [i for i, x in enumerate(ms_) if x <= 100]
late = [i for i, x in enumerate(ms_) if x > 100]
sc = a2.scatter(px, ry, c=ms_, cmap="viridis", s=70, zorder=3, label="4-task probe (colored by step)")
fig.colorbar(sc, ax=a2, label="step")
d8m = [r["step"] for r in d8 if r["step"] in vals]
a2.scatter([vals[x] for x in d8m], [next(r["pooled"] for r in d8 if r["step"] == x) for x in d8m],
           marker="D", s=110, facecolor="none", edgecolor="#c53030", lw=2, zorder=4, label="FULL val-8")
r_all, p_all = stats.pearsonr(px, ry)
rho_all, _ = stats.spearmanr(px, ry)
r_e, _ = stats.pearsonr(px[early], ry[early]) if len(early) > 2 else (float("nan"), 0)
r_l, _ = stats.pearsonr(px[late], ry[late]) if len(late) > 2 else (float("nan"), 0)
a2.set_title(f"matched steps: r={r_all:.2f} (ρ={rho_all:.2f}) | ≤100: r={r_e:.2f} | >100: r={r_l:.2f}")
a2.set_xlabel("proxy reward (C4b val)")
a2.set_ylabel("4-task rollout success %")
a2.legend(fontsize=8, loc="lower right")
a2.grid(alpha=0.2)

fig.suptitle("v7a — does the proxy reward track real rollout success?", fontsize=12)
fig.text(0.01, 0.01, "v7a reward operating point: C=1 (no context averaging), F=16 — fine-grid sign-acc at C=1 is ~57-61% on close pairs, "
         "so the proxy resolves the big early gains but not late fine improvements. Rollout n=192/pt (±3.6pp); proxy n=40 contexts.",
         fontsize=7, color="#4a5568")
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig("results/charts/v7a_proxy_vs_rollout.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
print(f"r_all={r_all:.3f} p={p_all:.4f}  rho={rho_all:.3f}  r_early={r_e:.3f}  r_late={r_l:.3f}")
print("chart -> results/charts/v7a_proxy_vs_rollout.png")
