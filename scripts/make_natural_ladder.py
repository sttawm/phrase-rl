#!/usr/bin/env python3
"""Natural-rephrase ladder, 24-LAYOUT BASIS (2026-08-06 conversion).
References use full 24-layout data. Dagger-marked treatment cells are layouts
0-11 only until their second halves land; they convert in place. Qwen arms
reinstated at END-OF-QUEUE priority (user). Pooled-forward triplet style."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_POOL, C_IV, C_OOV = "#a3bffa", "#b2f5ea", "#fed7aa"

# (letter, description, (pooled, IV, OOV)) — ALL 24 layouts
REFS = [("A", "original (canonical) — 24 layouts, 12 reps", (36.08, 48.3, 27.3)),
        ("B", "natural rephrases, no rewriter — 24 layouts", (28.45, 41.5, 19.2)),
        ("C", "natural on NON-augmented $\\pi_0$ — 24 layouts", (23.78, 35.6, 15.4)),
        ("D", "adversarial ERT, no rewriter — 24 layouts, 12 reps", (26.59, 30.6, 23.7))]

ROLES = ["rollout+training-data rules", "rollout-derived rules", "no rules (bare prompt)"]
# cell = (pooled, IV, OOV, is_partial_0_11) or None (pending)
MODELS = {
    "Frozen Qwen":  [(29.41, 35.2, 25.3, False), (30.44, 33.4, 28.3, False), (28.32, 39.1, 20.6, False)],
    "Gemini-Pro":   [(33.92, 42.0, 28.2, False), (36.89, 42.2, 33.1, False), (31.15, 43.6, 22.2, False)],
    "Claude Fable": [(31.18, 40.9, 24.3, False), (36.35, 41.2, 32.9, False), (30.75, 43.3, 21.8, False)],
}
PENDING = {}


def triplet_bars(ax, x, pool, iv, oov, dagger=False):
    ax.bar(x - 0.20, iv, 0.36, color=C_IV, alpha=0.55, zorder=1, edgecolor="none")
    ax.bar(x + 0.20, oov, 0.36, color=C_OOV, alpha=0.55, zorder=1, edgecolor="none")
    ax.text(x - 0.33, iv + 0.5, f"{iv:.0f}", ha="center", fontsize=6.6, color="#4a5568", zorder=3)
    ax.text(x + 0.33, oov + 0.5, f"{oov:.0f}", ha="center", fontsize=6.6, color="#4a5568", zorder=3)
    ax.bar(x, pool, 0.42, color=C_POOL, zorder=2, edgecolor="#4a5568", lw=0.9)
    lbl = f"{pool:.1f}" + ("$^\\dagger$" if dagger else "")
    ax.text(x, pool + 0.7, lbl, ha="center", fontsize=8.8, fontweight="bold", zorder=3)


fig, ax = plt.subplots(figsize=(15.8, 5.0))
x = 0.0
ticks, labels = [], []
for letter, _d, (p, iv, oo) in REFS:
    triplet_bars(ax, x, p, iv, oo)
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
            p, iv, oo, part = cell
            triplet_bars(ax, x, p, iv, oo, dagger=part)
        ax.text(x, -3.6, ROLES[i].replace("rules", "\nrules").replace(" (", "\n("), ha="center", va="top",
                fontsize=7.0, color="#4a5568")
        cxs.append(x)
        x += 1.35
    ticks.append(sum(cxs) / len(cxs))
    labels.append("\n\n\n\n" + model)
    x += 0.7
handles = [plt.Rectangle((0, 0), 1, 1, color=C_POOL),
           plt.Rectangle((0, 0), 1, 1, color=C_IV, alpha=0.55),
           plt.Rectangle((0, 0), 1, 1, color=C_OOV, alpha=0.55)]
ax.legend(handles, ["pooled (5 + 7)", "in-vocab (5)", "out-of-vocabulary (7)"],
          fontsize=8.5, loc="upper right")
ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=9.5)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("success % on natural rephrases")
ax.set_ylim(0, 52)
ax.grid(axis="y", alpha=0.25)
ax.set_title("Natural-rephrase ladder — 24-layout basis "
             "(executor: rephrase-augmented $\\pi_0$ unless noted)", fontsize=10.5, pad=12)
fig.text(0.01, 0.005,
         "Naturals: 1 rep/layout over all 24 layouts; references A and D: 12 reps.",
         fontsize=6.8, color="#4a5568")
fig.tight_layout()
fig.savefig("results/charts/natural_ladder.png", dpi=140, bbox_inches="tight", pad_inches=0.25)
print("chart -> results/charts/natural_ladder.png")
