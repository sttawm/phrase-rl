#!/usr/bin/env python3
"""Natural-rephrase ladder: rewriter treatments over the K=16 natural human
rephrases (layouts 0-11, executor = rephrase-augmented pi0), pooled-forward
style. Pending arms (rolling tonight) shown as hatched placeholders with ETAs.
Refs layout-matched from raw parquets; natural refs = the rephrases passed
through unmodified."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_POOL, C_IV, C_OOV = "#a3bffa", "#b2f5ea", "#fed7aa"

REFS = [("A", "original (canonical phrasing)", (34.38, 44.6, 27.1)),
        ("B", "natural rephrases, no rewriter (layouts 0-11)", (26.30, 38.6, 17.5)),
        ("C", "natural, no rewriter — ALL 24 layouts", (28.45, 41.5, 19.2)),
        ("D", "natural on NON-augmented $\\pi_0$ (24 layouts)", (23.78, 35.6, 15.4)),
        ("E", "adversarial ERT, no rewriter", (26.85, 31.5, 23.5))]

ROLES = ["rules-v4", "no rules (prompt B)"]
MODELS = {
    "Frozen Qwen":  [None, None],
    "Gemini-Pro":   [(34.33, 40.4, 30.0), None],
    "Claude Fable": [(30.43, 38.8, 24.4), None],
}
PENDING = {
    ("Frozen Qwen", 0): "rolling — ETA ~07:30",
    ("Frozen Qwen", 1): "rolling — ETA ~04:15",
    ("Gemini-Pro", 1): "rolling — ETA ~03:00",
    ("Claude Fable", 1): "rolling — ETA ~04:00",
}


def triplet_bars(ax, x, vals):
    pool, iv, oov = vals
    ax.bar(x - 0.20, iv, 0.36, color=C_IV, alpha=0.55, zorder=1, edgecolor="none")
    ax.bar(x + 0.20, oov, 0.36, color=C_OOV, alpha=0.55, zorder=1, edgecolor="none")
    ax.text(x - 0.33, iv + 0.5, f"{iv:.0f}", ha="center", fontsize=6.6, color="#4a5568", zorder=3)
    ax.text(x + 0.33, oov + 0.5, f"{oov:.0f}", ha="center", fontsize=6.6, color="#4a5568", zorder=3)
    ax.bar(x, pool, 0.42, color=C_POOL, zorder=2, edgecolor="#4a5568", lw=0.9)
    ax.text(x, pool + 0.7, f"{pool:.1f}", ha="center", fontsize=8.8, fontweight="bold", zorder=3)


fig, ax = plt.subplots(figsize=(13.8, 5.0))
x = 0.0
ticks, labels = [], []
for letter, _desc, vals in REFS:
    triplet_bars(ax, x, vals)
    ax.axvspan(x - 0.55, x + 0.55, color="#000000", alpha=0.04, zorder=0)
    ticks.append(x)
    labels.append("★ " + letter)
    x += 1.35
key = "\n".join(f"{l} — {d}" for l, d, _ in REFS)
ax.text(0.012, 0.975, key, transform=ax.transAxes, fontsize=7.6, va="top",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f7fafc", ec="#cbd5e0", alpha=0.95), zorder=5)
x += 0.55
for model, cells in MODELS.items():
    cxs = []
    for i, cell in enumerate(cells):
        if cell is None:
            ax.bar(x, 42, 0.76, color="none", edgecolor="#a0aec0", ls="--", lw=1.0,
                   hatch="//", alpha=0.30)
            ax.text(x, 2.2, PENDING[(model, i)], ha="center", va="bottom", fontsize=7.2,
                    color="#718096", style="italic", rotation=90)
        else:
            triplet_bars(ax, x, cell)
        ax.text(x, -3.4, ROLES[i].replace(" (", "\n("), ha="center", va="top",
                fontsize=7.2, color="#4a5568")
        cxs.append(x)
        x += 1.35
    ticks.append(sum(cxs) / len(cxs))
    labels.append("\n\n" + model)
    x += 0.7
handles = [plt.Rectangle((0, 0), 1, 1, color=C_POOL),
           plt.Rectangle((0, 0), 1, 1, color=C_IV, alpha=0.55),
           plt.Rectangle((0, 0), 1, 1, color=C_OOV, alpha=0.55)]
ax.legend(handles, ["pooled (5 + 7)", "in-vocab (5)", "out-of-vocabulary (7)"],
          fontsize=8.5, loc="upper right")
ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=9.5)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("success % on natural rephrases (layouts 0-11)")
ax.set_ylim(0, 46)
ax.grid(axis="y", alpha=0.25)
ax.set_title("Natural-rephrase ladder — rewriter treatments over K=16 human rephrasings "
             "(executor: rephrase-augmented $\\pi_0$ unless noted)", fontsize=10.5, pad=12)
fig.text(0.01, 0.005,
         "treatment arms: layouts 0-11, 1 rep (their reference = the 0-11 natural column). 24-layout halves: "
         "augmented 26.3 / 30.6, non-augmented 25.0 / 22.6 — augmentation gap 4.7 at full coverage. Refs: 12 reps except naturals.",
         fontsize=6.8, color="#4a5568")
fig.tight_layout()
fig.savefig("results/charts/natural_ladder.png", dpi=140, bbox_inches="tight", pad_inches=0.25)
print("chart -> results/charts/natural_ladder.png")
