#!/usr/bin/env python3
"""pi0 input-condition chart, CURRENT sealed data. Main sequence (executor =
rephrase-augmented INTACT pi0): oracle / canonical / VLM-generated naturals
(the K=16 robustness set, 192 phrases — NOT the 186-phrase main-eval set;
caption footnote covers this) / adversarial / human-generated naturals on the
12 sealed tasks x 24 pinned layouts. Right of the divider: the same K=16
naturals passed through the PLAIN executor (INTACT-pi0-finetune-bridge, no
rephrase augmentation; Amendments 19/24, 24 layouts x 1 rep) — the only
condition ever rolled on that checkpoint. Pooled bar solid in
front (white value); in-distribution (5 tasks) / out-of-distribution (7)
strata behind, thinner and semi-transparent. No in-image title.

Provenance (verified 2026-09-13, base-weighted):
  oracle      48.2 / 50.3 / 46.8   results/sealed/oracle_confirmed_x12.parquet (24x12)
  canonical   36.1 / 48.3 / 27.3   anchors_x12 originals (24x12)
  adversarial 24.5 / 25.2 / 23.9   a29pass legs + anchors passthrough (72 attacks)
  human nat   23.0 / 29.6 / 18.5   a39_human_cells.json raw_human (363 pairs, 24x1)
  K16 pair    from rephrase_robustness.jsonl (both layout halves; 16/task
              balanced so task-weighted == base-weighted)
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
hm = json.loads((R / "results/analysis/a39_human_cells.json").read_text())["raw_human"]
rr = {r["arm"]: r for r in map(json.loads,
      open(R / "results/analysis/rephrase_robustness.jsonl"))}

IV = {"widowx_carrot_on_sponge_clean", "widowx_eggplant_on_sponge_clean",
      "widowx_cube_on_plate_clean", "widowx_nut_on_plate_clean",
      "widowx_small_plate_on_green_cube_clean"}


def k16(exe):
    h1, h2 = rr[f"rephrase16_{exe}"], rr[f"rephrase16_{exe}_lay12"]
    pt = {t: (h1["per_task"][t] + h2["per_task"][t]) / 2 for t in h1["per_task"]}
    iv = [v for t, v in pt.items() if t in IV]
    oov = [v for t, v in pt.items() if t not in IV]
    return (sum(pt.values()) / len(pt), sum(iv) / len(iv), sum(oov) / len(oov))


C_ORACLE, C_CANON = "#3F6B52", "#6B5E9B"
C_NAT, C_ADV, C_HUM = "#B08A3E", "#A94E4E", "#5E7B9B"
C_PLAIN, INK = "#7A8698", "#2d3748"

MAIN = [
    ("oracle\n(searched)",        (48.23, 50.28, 46.78), C_ORACLE),
    ("canonical\ninstruction",    (36.08, 48.33, 27.33), C_CANON),
    ("VLM-generated\nnaturals",   k16("pi0rephrase"), C_NAT),
    ("adversarial",               (24.46, 25.17, 23.94), C_ADV),
    ("human-generated\nnaturals", (hm["pooled"], hm["iv"], hm["oov"]), C_HUM),
]
PAIR = [
    ("VLM naturals,\nplain $\\pi_0$",      k16("pi0base"),     C_PLAIN),
]

fig, ax = plt.subplots(figsize=(9.0, 4.2))


def cluster(x, pool, iv, oov, col):
    ax.bar(x - 0.20, iv, 0.34, color=col, alpha=0.40, zorder=1, edgecolor="none")
    ax.bar(x + 0.20, oov, 0.34, color=col, alpha=0.40, zorder=1, edgecolor="none")
    ax.text(x - 0.32, iv + 0.7, f"{iv:.0f}", ha="center", fontsize=7,
            color="#4a5568", zorder=3)
    ax.text(x + 0.32, oov + 0.7, f"{oov:.0f}", ha="center", fontsize=7,
            color="#4a5568", zorder=3)
    ax.bar(x, pool, 0.44, color=col, zorder=2, edgecolor="none")
    ax.text(x, pool - 1.2, f"{pool:.1f}", ha="center", va="top", fontsize=9.5,
            fontweight="bold", color="white", zorder=4,
            bbox=dict(boxstyle="square,pad=0.12", fc=col, ec="none"))


x, ticks, labels = 0.0, [], []
for name, (pool, iv, oov), col in MAIN:
    cluster(x, pool, iv, oov, col)
    ticks.append(x); labels.append(name); x += 1.05
div = x - 0.18
ax.axvline(div, color="#cbd5e0", lw=1.0, ls=(0, (2, 2)), zorder=0)
x += 0.30
for name, (pool, iv, oov), col in PAIR:
    cluster(x, pool, iv, oov, col)
    ticks.append(x); labels.append(name); x += 1.05

ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=8.4)
ax.set_ylabel("success % (12 sealed tasks, 24 layouts)", fontsize=9.5)
ax.set_ylim(0, 55)
ax.set_xlim(-0.62, x - 0.45)
ax.grid(axis="y", alpha=0.18, zorder=0)
ax.tick_params(axis="x", length=0)
ax.spines[["top", "right"]].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, fc="#7A8698"),
           plt.Rectangle((0, 0), 1, 1, fc="#7A8698", alpha=0.40)]
ax.legend(handles, ["pooled (12 tasks)",
                    "flanks: in-distribution (left, 5) / out-of-distribution (right, 7)"],
          fontsize=7.8, loc="upper right", frameon=False)

fig.tight_layout()
out = R / "results/charts/pi0_conditions.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
print("chart ->", out)
