#!/usr/bin/env python3
"""Sealed ladders, one chart per condition, vertical grouped bars.

Left group: references (oracle / original / adversarial passthrough).
Then one 3-bar cluster per model: train+rollout rules (v4), rollout rules (v3),
no rules (bare). Missing cells render as dashed hatched placeholders.
Pooled sealed values (12 tasks x 24 x 12)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REFS = [("oracle*", 48.23, "#e2b8b8"), ("original", 36.08, "#b8d8c8"),
        ("adversarial\n(pass through)", 26.59, "#cbd5e0")]

ROLE = [("train+rollout rules (v4)", "#a3bffa"), ("rollout rules (v3)", "#b2f5ea"),
        ("no rules (bare)", "#fed7aa")]

DATA = {
    "adversarial": {
        "Frozen Qwen": [30.06, 31.48, 30.96],
        "Gemini-Pro": [33.30, 31.57, 27.78],
        "Claude Fable": [34.26, 30.99, None],  # bare rolling (arm H)
    },
    "nominal": {
        "Frozen Qwen": [27.81, None, None],
        "Gemini-Pro": [37.18, 34.78, None],
        "Claude Fable": [37.73, None, None],
    },
}
PENDING_NOTE = {("adversarial", "Claude Fable", 2): "rolling"}

for cond, models in DATA.items():
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    x = 0.0
    ticks, ticklabels = [], []
    for name, val, col in REFS:
        ax.bar(x, val, 0.8, color=col, edgecolor="#4a5568", lw=0.8)
        ax.text(x, val + 0.6, f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")
        ticks.append(x)
        ticklabels.append("★ " + name)
        x += 1.0
    x += 0.9  # gap after references
    first = True
    for model, vals in models.items():
        cx = []
        for i, ((role, col), val) in enumerate(zip(ROLE, vals)):
            if val is None:
                ax.bar(x, 50, 0.8, color="none", edgecolor="#a0aec0", ls="--", lw=1.0,
                       hatch="//", alpha=0.30)
                note = PENDING_NOTE.get((cond, model, i), "not yet run")
                ax.text(x, 2.5, note, ha="center", va="bottom", fontsize=7.5,
                        color="#718096", style="italic", rotation=90)
            else:
                ax.bar(x, val, 0.8, color=col, edgecolor="none",
                       label=role if first else None)
                ax.text(x, val + 0.6, f"{val:.1f}", ha="center", fontsize=9)
            cx.append(x)
            x += 1.0
        first = False
        ticks.append(sum(cx) / len(cx))
        ticklabels.append(model)
        x += 0.9  # gap between clusters
    # legend needs all three roles even if first cluster had a placeholder
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in ROLE]
    ax.legend(handles, [r for r, _ in ROLE], fontsize=8.5, loc="upper right")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=9.5)
    ax.set_ylabel("sealed success % (12 tasks × 24 × 12)")
    ax.set_ylim(0, 55)
    ax.grid(axis="y", alpha=0.25)
    title_cond = ("ADVERSARIAL input (repair)" if cond == "adversarial"
                  else "ORIGINAL input (polish)")
    ax.set_title(f"Sealed ladder — {title_cond}    "
                 "(★ reference | *oracle phrases selected on layouts 0-17)", fontsize=10.5)
    fig.tight_layout()
    out = f"results/charts/sealed_ladder_{cond}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.25)
    print("chart ->", out)
