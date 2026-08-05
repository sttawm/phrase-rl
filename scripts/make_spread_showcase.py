#!/usr/bin/env python3
"""Phrase-spread showcases: for 4 tasks (2 in-vocab, 2 out-of-vocabulary),
five nearly-identical phrasings and their rollout success. Oracle on top.
Values: layouts 0-11 (oracle/original/adversarial = 144 eps; naturals = 24)."""
import json

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rr = {r["arm"]: r for r in map(json.loads, open("results/analysis/rephrase_robustness.jsonl"))}
NAT = rr["rephrase16_pi0rephrase"]["per_rephrase"]
a = pd.read_parquet("results/sealed/anchors_x12.parquet"); a = a[a.episode_id < 12]
o = pd.read_parquet("results/sealed/oracle_confirmed_x12.parquet"); o = o[o.episode_id < 12]

C = {"oracle": "#b2f5ea", "original": "#a3bffa", "natural": "#fed7aa", "adversarial": "#feb2b2"}

# task -> (stratum, [natural phrases to include])
PICKS = {
    "widowx_nut_on_plate_clean": ("in-vocab", ["Drop the nut onto the plate.", "Grab the nut and place it on the plate."]),
    "widowx_carrot_on_sponge_clean": ("in-vocab", ["Move the carrot onto the sponge.", "Transfer the carrot to the top of the sponge."]),
    "widowx_coke_can_on_keyboard_clean": ("out-of-vocabulary", ["Set the Coca-Cola can on the keyboard.", "Move the red soda can onto the keyboard."]),
    "widowx_pepsi_on_plate_clean": ("out-of-vocabulary", ["Put the blue soda can onto the plate.", "Move the Pepsi can over to the plate."]),
}

for task, (stratum, nat_picks) in PICKS.items():
    rows = []
    op = o[o.task == task]
    rows.append((op.phrase.iloc[0], op.success.mean() * 100, "oracle"))
    orig = a[(a.arm == "originals") & (a.task == task)]
    rows.append((orig.phrase.iloc[0], orig.success.mean() * 100, "original"))
    ert = a[(a.arm == "passthrough") & (a.task == task)]
    rows.append((ert.phrase.iloc[0], ert.success.mean() * 100, "adversarial"))
    for p in nat_picks:
        rows.append((p, NAT[f"{task}|{p}"], "natural"))
    rows.sort(key=lambda r: r[1], reverse=True)

    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    ys = range(len(rows))[::-1]
    for y, (phrase, val, cat) in zip(ys, rows):
        ax.barh(y, val, 0.62, color=C[cat], edgecolor="#4a5568", lw=0.7)
        label = f'"{phrase}"'
        ax.text(0.8, y, label, va="center", fontsize=9, color="#1a202c")
        ax.text(val + 1.0, y, f"{val:.0f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([cat for _, _, cat in rows], fontsize=8.5, color="#4a5568")
    ax.set_xlim(0, max(v for _, v, _ in rows) + 12)
    ax.set_xlabel("rollout success % (layouts 0-11; oracle/original/adversarial 144 eps, naturals 24)")
    ax.grid(axis="x", alpha=0.25)
    short = task[7:-6]
    ax.set_title(f"Same task, nearly-identical requests — {short} ({stratum})", fontsize=11)
    fig.tight_layout()
    out = f"results/charts/spread_{short}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.2)
    print("chart ->", out)
