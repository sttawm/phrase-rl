#!/usr/bin/env python3
"""results/charts/results_translating_v2.png -- Fig-9 v2: identical layout, with
the ADVERSARIAL condition re-measured at widened coverage (A29: 6 attacks/task,
72 phrases, 24 layouts x 2 reps) for the rollout-derived (sim-loop) books and
the no-rephraser adversarial baseline. Cells NOT re-run at 6/task keep their
original-12 numbers and are marked with a dagger.

Layout: a narrow leftmost BASELINES block (the four no-rephraser references,
drawn once) + three condition panels (Adversarial / Original / Natural inputs)
x three appliers (Gemini, Claude, Qwen) x three rule conditions.
Baseline levels also run across every panel as dotted lines. All baseline
pooled bars -- including the bare-prompt (no rules) cells -- are grey; rules
cells are blue. Replaces the natural ladder, the adversarial ladder, and the
rules-ablation chart in the paper.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = [("oracle*", 48.2, 50.3, 46.8, "#2f855a"),
        ("original", 36.1, 48.3, 27.3, "#4a5568"),
        ("natural", 28.45, 41.5, 19.2, "#b7791f"),
        ("adversarial", 24.5, 25.2, 23.9, "#c53030"),   # v2: 72-attack passthrough
        ("natural,\nstd. $\\pi_0$", 23.8, 35.6, 15.4, "#805ad5")]

ROLES = ["rollout+\ntraining-data\nrules", "rollout-\nderived\nrules", "no rules\n(bare\nprompt)"]
MODELS = ["Gemini-Pro", "Claude Fable", "Frozen Qwen"]

DATA = {
    "Adversarial": {
        "Gemini-Pro":   [(33.3, 37.0, 30.7), (31.7, 38.0, 27.1), (27.8, 33.1, 24.0)],
        "Claude Fable": [(34.3, 40.6, 29.7), (29.3, 35.8, 24.6), (28.0, 36.5, 21.9)],
        "Frozen Qwen":  [(30.1, 35.3, 26.3), (25.4, 30.8, 21.5), (29.6, 40.5, 21.9)],
    },
    "Original": {
        "Gemini-Pro":   [(37.2, 46.0, 30.9), (34.8, 42.1, 29.6), None],
        "Claude Fable": [(37.7, 47.6, 30.7), (34.7, 41.9, 29.6), None],
        "Frozen Qwen":  [(27.8, 34.9, 22.7), (32.2, 34.9, 30.3), (29.8, 40.7, 22.0)],
    },
    "Natural": {
        "Gemini-Pro":   [(33.9, 42.0, 28.2), (36.9, 42.2, 33.1), (31.2, 43.6, 22.2)],
        "Claude Fable": [(31.2, 40.9, 24.3), (36.4, 41.2, 32.9), (30.8, 43.3, 21.8)],
        "Frozen Qwen":  [(29.4, 35.2, 25.3), (30.4, 33.4, 28.3), (28.3, 39.1, 20.6)],
    },
}
C_RULES, C_BASEBAR = "#a3bffa", "#8b96a5"     # blue = rules cells, grey = baselines
C_IV, C_OOV = "#b2f5ea", "#fed7aa"
YMIN, YMAX = 15, 52     # floor set by the std-pi0 natural OOV stratum (15.4)

fig, axes = plt.subplots(1, 4, figsize=(21.8, 5.8), sharey=True,
                         gridspec_kw={"width_ratios": [1.75, 3, 3, 2.15]})


def triplet(ax, x, p, iv, oo, pooled_color):
    ax.bar(x - 0.20, iv, 0.34, color=C_IV, alpha=0.55, zorder=2)
    ax.bar(x + 0.20, oo, 0.34, color=C_OOV, alpha=0.55, zorder=2)
    ax.text(x - 0.32, iv + 0.4, f"{iv:.0f}", ha="center", fontsize=6.6,
            color="#4a5568", zorder=4)
    ax.text(x + 0.32, oo + 0.4, f"{oo:.0f}", ha="center", fontsize=6.6,
            color="#4a5568", zorder=4)
    ax.bar(x, p, 0.42, color=pooled_color, zorder=3, edgecolor="#4a5568", lw=0.9)
    ax.text(x, p + 0.55, f"{p:.1f}", ha="center", fontsize=8.6,
            fontweight="bold", zorder=4)


# ---------------- baselines block (drawn once) ----------------
axB = axes[0]
for i, (name, p, iv, oo, c) in enumerate(BASE):
    triplet(axB, float(i), p, iv, oo, C_BASEBAR)
    axB.text(i, YMIN - 1.1, name.replace("adversarial", "adver-\nsarial").replace("oracle*", "oracle$^{*}$"),
             ha="center", va="top", fontsize=7.2, color="#4a5568")
axB.set_title("Baselines\n(no rephraser)", fontsize=11.5, pad=10)
axB.set_xlim(-0.7, len(BASE) - 0.3)
axB.set_xticks([])
axB.set_ylabel("sealed-suite success %", fontsize=11)

# ---------------- condition panels ----------------
for ax, cond in zip(axes[1:], ["Adversarial", "Natural", "Original"]):
    x = 0.0
    ticks, tlabels = [], []
    for mi, model in enumerate(MODELS):
        cxs = []
        cells = DATA[cond][model]
        if cond == "Original":
            cells = cells[:2]          # bare-prompt column dropped for Originals
        for ri, cell in enumerate(cells):
            p, iv, oo = cell
            color = C_BASEBAR if ri == 2 else C_RULES       # bare prompt = baseline grey
            triplet(ax, x, p, iv, oo, color)
            ax.text(x, YMIN - 1.1, ROLES[ri], ha="center", va="top", fontsize=6.4,
                    color="#4a5568")
            cxs.append(x)
            x += 1.18
        ticks.append(sum(cxs) / len(cxs))
        tlabels.append(model)
        if mi < len(MODELS) - 1:
            ax.axvline(x - 0.28, color="#cbd5e0", lw=1.1, zorder=1)
            x += 0.50
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabels, fontsize=10.5)
    ax.tick_params(axis="x", pad=44, length=0)
    ax.set_xlim(-0.75, x - 0.80)
    ax.grid(axis="y", alpha=0.18)
    ax.set_title("Translating $\\bf{%s}$ Task Instructions" % cond,
                 fontsize=12.5, pad=10)

# dotted baseline levels across every panel
for ax in axes:
    for name, p, iv, oo, c in BASE:
        ax.axhline(p, color=c, lw=1.2, ls=(0, (3, 3)), zorder=1, alpha=0.85)
axes[0].set_ylim(YMIN, YMAX)

# right-edge line labels
axR = axes[3]
for name, p, iv, oo, c in BASE:
    axR.text(axR.get_xlim()[1] + 0.06, p, name.rstrip("*").replace(",\n", " "), fontsize=8.2,
             color=c, va="center", ha="left", clip_on=False)

handles = [plt.Rectangle((0, 0), 1, 1, color=C_RULES),
           plt.Rectangle((0, 0), 1, 1, color=C_BASEBAR),
           plt.Rectangle((0, 0), 1, 1, color=C_IV, alpha=0.55),
           plt.Rectangle((0, 0), 1, 1, color=C_OOV, alpha=0.55),
           plt.Line2D([0], [0], color="#4a5568", lw=1.2, ls=(0, (3, 3)))]
axes[1].legend(handles, ["rules cell (pooled)", "baseline (pooled)",
                         "in-distribution (5)", "out-of-distribution (7)", "baseline level"],
               fontsize=7.4, loc="upper left", framealpha=0.95)

fig.tight_layout(rect=(0, 0, 0.985, 1.0))
fig.text(0.985, 0.015,
         "v2: adversarial baseline + rollout-derived-rules cells re-measured on the widened attack set "
         "(6 attacks/task, n=3,456/arm, 24 layouts x 2 reps); "
         "training-data-rules and bare-prompt adversarial cells$^\\dagger$ retain original single-attack coverage (re-runs in flight).",
         ha="right", fontsize=7.4, color="#4a5568")
fig.savefig("results/charts/results_translating_v2.png", dpi=140, bbox_inches="tight",
            pad_inches=0.25)
print("chart -> results/charts/results_translating_v2.png")
