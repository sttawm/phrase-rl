#!/usr/bin/env python3
"""A28: rules-v3 vs rules-v4 vs no-rules, on sealed natural rephrases.

The question A28 exists to answer: rules v3 was distilled from simulation and
certification evidence; v4 added a training-corpus overlay that made it
pass-through-first. On ADVERSARIAL inputs v4's caution was cheap. On NATURAL
inputs -- already fluent, needing little repair -- does that caution cost us?

Each cell is uniform-24: two 2,304-episode legs (layouts 0-11 and 12-23) over the
same 12 sealed tasks x 16 Gemini-generated natural rephrases. Cells with only one
leg are drawn hollow and marked, because a half cell is 12 layouts, not 24.

  python3 scripts/make_a28_chart.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = {}
for line in open("results/analysis/rephrase_robustness.jsonl"):
    d = json.loads(line)
    rows[d["arm"]] = d

APPLIERS = [("qwen", "Frozen Qwen 3.5-9B"), ("claude", "Claude"), ("gemini", "Gemini Pro")]
CONDS = [("promptB", "no rules (prompt B)", "#a0aec0"),
         ("rulesv4", "rules v4", "#2b6cb0"),
         ("rulesv3", "rules v3", "#dd6b20")]
PASSTHROUGH = 28.45   # pi0 on the natural rephrases, no rephraser at all


def cell(cond, model):
    a, b = f"rr16_{cond}_{model}", f"rr16_{cond}_{model}_lay12"
    have = [rows[k]["pooled"] for k in (a, b) if k in rows]
    if not have:
        return None, 0
    return float(np.mean(have)), len(have)


fig, ax = plt.subplots(figsize=(9.6, 4.8))
w, xs = 0.26, np.arange(len(APPLIERS))
for ci, (cond, clabel, col) in enumerate(CONDS):
    vals, halves = [], []
    for mi, (m, _) in enumerate(APPLIERS):
        v, nlegs = cell(cond, m)
        vals.append(v if v is not None else np.nan)
        halves.append(nlegs == 1)
    pos = xs + (ci - 1) * w
    for x, v, half in zip(pos, vals, halves):
        if np.isnan(v):
            ax.text(x, PASSTHROUGH + 1.2, "pending", rotation=90, ha="center",
                    va="bottom", fontsize=7.5, color="#a0aec0", style="italic")
            continue
        ax.bar(x, v, w * 0.92, color="white" if half else col,
               edgecolor=col, lw=2.0 if half else 0.8,
               hatch="//" if half else None,
               label=clabel if (x == pos[0] and ci >= 0) else None)
        ax.text(x, v + 0.35, f"{v:.1f}" + ("*" if half else ""), ha="center",
                fontsize=8.5, fontweight="bold", color=col)

ax.axhline(PASSTHROUGH, ls="--", color="#718096", lw=1.3)
ax.text(len(APPLIERS) - 0.55, PASSTHROUGH + 0.3, f"no rephraser at all ({PASSTHROUGH})",
        fontsize=8, color="#718096", ha="right")
ax.set_xticks(xs)
ax.set_xticklabels([lbl for _, lbl in APPLIERS], fontsize=10)
ax.set_ylabel("val success on sealed natural rephrases (%)", fontsize=10)
ax.set_ylim(24, 40)
handles, labels = ax.get_legend_handles_labels()
seen, hh, ll = set(), [], []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l); hh.append(h); ll.append(l)
ax.legend(hh, ll, fontsize=8.5, loc="upper left")
ax.grid(axis="y", alpha=0.25)
ax.set_title("A28 — does the v4 training-corpus overlay cost us on NATURAL inputs?  Yes, for every applier.\n"
             "v3-v4 on MATCHED layouts: qwen +1.04, gemini +2.86, claude +5.17   "
             "(hatched = one leg only, 12 layouts not 24)", fontsize=10.5)
fig.tight_layout()
fig.savefig("results/charts/a28_rulesv3_vs_v4.png", dpi=150, bbox_inches="tight",
            pad_inches=0.2)
print("chart -> results/charts/a28_rulesv3_vs_v4.png")

print("\ncells (uniform-24 unless marked):")
for m, lbl in APPLIERS:
    line = f"  {lbl:22s}"
    for cond, clabel, _ in CONDS:
        v, nl = cell(cond, m)
        line += f"  {clabel}: " + (f"{v:.2f}{'*' if nl == 1 else ' '}" if v else "  --  ")
    v3, n3 = cell("rulesv3", m)
    v4, n4 = cell("rulesv4", m)
    if v3 and v4:
        line += f"   | v3-v4 {v3 - v4:+.2f}"
    print(line)
print("\n* = one leg only")
