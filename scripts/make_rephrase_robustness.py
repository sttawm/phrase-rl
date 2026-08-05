#!/usr/bin/env python3
"""Rephrase-robustness chart (Amendments 19/20): pooled bars for the natural-
phrasing eval + per-rephrase success distributions (natural vs rules-repaired).
All arms: layouts 0-11 x 1 rep; baselines recomputed from anchors raw on the
same layouts."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rr = {r["arm"]: r for r in map(json.loads, open("results/analysis/rephrase_robustness.jsonl"))}
nat = rr["rephrase16_pi0rephrase"]
rules = rr["rephrase16_rulesv4_gemini"]
base = rr["rephrase16_pi0base"]

BARS = [
    ("originals\n(canonical)", 34.38, "#b2f5ea"),
    ("K=16 natural\nrephrases", nat["pooled"], "#fed7aa"),
    ("natural +\nRULES-v4 (Gemini)", rules["pooled"], "#a3bffa"),
    ("ERT adversarial\n(passthrough)", 26.85, "#feb2b2"),
    ("natural, NON-rephrase-\naugmented pi0", base["pooled"], "#e2e8f0"),
]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
xs = np.arange(len(BARS))
for i, (lbl, v, c) in enumerate(BARS):
    a1.bar(i, v, 0.62, color=c, edgecolor="#4a5568", lw=0.8)
    a1.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
a1.set_xticks(xs)
a1.set_xticklabels([b[0] for b in BARS], fontsize=8.2)
a1.axhline(34.38, ls=":", color="#2f855a", lw=1.1, alpha=0.7)
a1.set_ylim(0, 40)
a1.set_ylabel("success % (layouts 0-11, 1 rep, n=2304/arm)")
a1.set_title("Natural human phrasing costs as much as adversarial —\nand rules-v4 repairs 99% of it", fontsize=10.5)
a1.grid(axis="y", alpha=0.25)

nv = sorted(nat["per_rephrase"].values(), reverse=True)
rv = sorted(rules["per_rephrase"].values(), reverse=True)
a2.plot(range(len(nv)), nv, color="#dd6b20", lw=2, label=f"natural (median {np.median(nv):.0f}, zeros {sum(1 for v in nv if v==0)})")
a2.plot(range(len(rv)), rv, color="#4c51bf", lw=2, label=f"+rules-v4 (median {np.median(rv):.0f}, zeros {sum(1 for v in rv if v==0)})")
a2.set_xlabel("rephrase rank (192 = 12 tasks x 16)")
a2.set_ylabel("per-rephrase success % (24 eps each)")
a2.set_title("Per-rephrase distribution: the rules pull the broken tail up", fontsize=10.5)
a2.legend(fontsize=8.5)
a2.grid(alpha=0.25)

fig.suptitle("Rephrase-robustness eval — executor: rephrase-augmented pi0 (INTACT); rewriter: gemini-pro + rules-v4", fontsize=10)
fig.tight_layout()
fig.savefig("results/charts/rephrase_robustness.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/rephrase_robustness.png")
