#!/usr/bin/env python3
"""Does the calibrated proxy rank phrases the way real rollouts do?

254 sim phrases carry BOTH a freshly measured proxy (verifier + gripper, F=4
C<=16) and a real rollout success rate. Pooled agreement looks strong, but the
rules loop never compares across tasks -- it picks between phrasings OF ONE
TASK. Split the two apart, because they give different answers.

  python3 scripts/make_proxy_validity_chart.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRATCH = ("/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/"
           "a45839fa-48ee-4aea-bdee-72eb4fc9dccf/scratchpad")
m = pd.read_parquet(f"{SCRATCH}/proxy_vs_truth.parquet")

rows = []
for t, g in m.groupby("task"):
    if len(g) < 5 or g.truth.nunique() < 2:
        continue
    rho = stats.spearmanr(g.proxy, g.truth)[0]
    rows.append({"task": t.replace("widowx_", ""), "n": len(g), "rho": rho,
                 "se": 1 / np.sqrt(len(g) - 1)})
w = pd.DataFrame(rows).sort_values("rho")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.2),
                               gridspec_kw={"width_ratios": [1.05, 1]})

for t, g in m.groupby("task"):
    axL.scatter(g.proxy, g.truth, s=15, alpha=0.6, label=t.replace("widowx_", ""))
pe = stats.pearsonr(m.proxy, m.truth)
sp = stats.spearmanr(m.proxy, m.truth)
lims = [-2, 102]
axL.plot(lims, lims, ls="--", color="#a0aec0", lw=1.2)
axL.set_xlim(lims); axL.set_ylim(lims)
axL.set_xlabel("calibrated proxy (%)", fontsize=10)
axL.set_ylabel("real rollout success (%)", fontsize=10)
axL.set_title(f"POOLED across tasks — looks strong\n"
              f"Pearson r={pe[0]:.2f}, Spearman ρ={sp[0]:.2f}, n={len(m)}",
              fontsize=10.5)
axL.legend(fontsize=6.4, loc="upper left", ncol=2)
axL.grid(alpha=0.22)

cols = ["#c53030" if r < 0 else ("#dd6b20" if r < 0.4 else "#2f855a") for r in w.rho]
y = np.arange(len(w))
axR.barh(y, w.rho, color=cols, height=0.62)
axR.errorbar(w.rho, y, xerr=w.se, fmt="none", ecolor="#4a5568", lw=1.0, capsize=2.5)
axR.axvline(0, color="#2d3748", lw=1.0)
axR.axvline(w.rho.mean(), color="#2b6cb0", ls="--", lw=1.5)
axR.text(w.rho.mean() + 0.02, len(w) - 0.6, f"mean {w.rho.mean():.2f}",
         color="#2b6cb0", fontsize=8.5, fontweight="bold")
axR.set_yticks(y)
axR.set_yticklabels([f"{r.task}  (n={r.n})" for r in w.itertuples()], fontsize=7.8)
axR.set_xlabel("within-task Spearman ρ vs real rollout success", fontsize=10)
axR.set_xlim(-0.75, 1.0)
axR.set_title("WITHIN task — what the loop actually uses\n"
              "bars ±1 s.e.; one task ranks backwards", fontsize=10.5)
axR.grid(axis="x", alpha=0.22)

fig.suptitle("The proxy separates EASY TASKS from hard ones far better than it "
             "separates GOOD PHRASINGS from bad ones on the same task",
             fontsize=11.5, y=0.995)
fig.tight_layout()
fig.savefig("results/charts/proxy_validity.png", dpi=155)
print("chart -> results/charts/proxy_validity.png")
print(w.round(3).to_string(index=False))
print(f"\nmean within-task rho {w.rho.mean():.3f}  (s.e. of the mean "
      f"{w.rho.std(ddof=1)/np.sqrt(len(w)):.3f})")
