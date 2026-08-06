#!/usr/bin/env python3
"""Rules x executor x input-distribution interaction (2026-08-06).
Panel A: repair fraction crossover between input distributions.
Panel B: the broken-tail mechanism — how each model treats the ~50 natural
phrasings that score 0, and what its rewrites achieve."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_CLAUDE, C_GEM = "#feb2b2", "#a3bffa"

fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})

# A: repair fraction (% of the no-rewriter -> originals gap closed), per distribution
xs = [0, 1]
claude = [81, 51]
gemini = [71, 99]
a.plot(xs, claude, "-o", color=C_CLAUDE, lw=2.5, ms=9, label="Claude Fable + rules-v4",
       markeredgecolor="#4a5568")
a.plot(xs, gemini, "-o", color=C_GEM, lw=2.5, ms=9, label="Gemini-Pro + rules-v4",
       markeredgecolor="#4a5568")
for x, v in zip(xs, claude):
    a.text(x, v - 6, f"{v}%", ha="center", fontsize=10, fontweight="bold", color="#c53030")
for x, v in zip(xs, gemini):
    a.text(x, v + 3.5, f"{v}%", ha="center", fontsize=10, fontweight="bold", color="#4c51bf")
a.set_xticks(xs)
a.set_xticklabels(["adversarial ERT\n(sealed, 24x12)", "natural rephrases\n(layouts 0-11, x1)"], fontsize=9)
a.set_xlim(-0.35, 1.35)
a.set_ylim(35, 112)
a.set_ylabel("repair fraction — % of phrasing damage recovered")
a.set_title("Same rules document, opposite ordering\nby input distribution", fontsize=10.5)
a.legend(fontsize=8.5, loc="lower left")
a.grid(axis="y", alpha=0.25)

# B: broken-tail mechanism (the 50 natural phrasings scoring 0 before rules)
groups = ["phrasings materially\nrewritten (of 50)", "success AFTER a\nmaterial rewrite (%)",
          "case/punct-only rate,\nall 192 inputs (%)"]
gvals = [40, 18.7, 44]
cvals = [36, 9.2, 57]
xg = range(len(groups))
for i, (gv, cv) in enumerate(zip(gvals, cvals)):
    b.bar(i - 0.18, gv, 0.34, color=C_GEM, edgecolor="#4a5568", lw=0.7)
    b.bar(i + 0.18, cv, 0.34, color=C_CLAUDE, edgecolor="#4a5568", lw=0.7)
    b.text(i - 0.18, gv + 1, f"{gv:g}", ha="center", fontsize=9, fontweight="bold")
    b.text(i + 0.18, cv + 1, f"{cv:g}", ha="center", fontsize=9, fontweight="bold")
b.set_xticks(list(xg))
b.set_xticklabels(groups, fontsize=8.3)
b.set_ylim(0, 66)
b.set_title("Mechanism on the broken tail: both models rewrite —\nGemini's rewrites work twice as well", fontsize=10.5)
b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_GEM),
                  plt.Rectangle((0, 0), 1, 1, color=C_CLAUDE)],
         labels=["Gemini-Pro", "Claude Fable"], fontsize=8.5)
b.grid(axis="y", alpha=0.25)

fig.suptitle("rules-v4 x executor x input distribution — the rules' effect is a property of the (document, applier, register) triple",
             fontsize=10.5)
fig.tight_layout()
fig.savefig("results/charts/rules_interaction.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/rules_interaction.png")
