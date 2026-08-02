#!/usr/bin/env python3
"""Sealed ladders, one chart per condition, vertical bars, each treatment shown
as a pooled / in-vocab / OOV triplet. References (oracle / original /
adversarial passthrough) lead; then one cluster per model. Nominal chart has no
bare slots (user: originals ARE the human baseline). Strata values from the
sealed per-task tables (audit stratification; 5 in-vocab / 7 OOV tasks)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (pooled, in_vocab, oov)
REFS = [("oracle*", (48.2, 50.3, 46.8)), ("original", (36.1, 48.3, 27.3)),
        ("adversarial\n(pass through)", (26.6, 30.6, 23.7))]

DATA = {
    "adversarial": {
        "roles": ["train+rollout rules (v4)", "rollout rules (v3)", "no rules (bare)"],
        "models": {
            "Frozen Qwen": [(30.1, 35.3, 26.3), (31.5, 35.7, 28.5), (24.2, 31.2, 19.2)],
            "Gemini-Pro": [(33.3, 37.0, 30.7), (31.6, 38.2, 26.8), (27.8, 40.2, 18.9)],
            "Claude Fable": [(34.3, 40.6, 29.7), (31.0, 35.3, 27.9), (29.1, 43.5, 18.9)],
        },
        "pending": {},
    },
    "nominal": {
        "roles": ["train+rollout rules (v4)", "rollout rules (v3)"],
        "models": {
            "Frozen Qwen": [(27.8, 34.9, 22.7), None],
            "Gemini-Pro": [(37.2, 46.0, 30.9), (34.8, 42.1, 29.6)],
            "Claude Fable": [(37.7, 47.6, 30.7), (34.7, 41.9, 29.6)],
        },
        "pending": {},
    },
}
STRATA = [("pooled", "#a3bffa"), ("in-vocab (5)", "#b2f5ea"), ("OOV (7)", "#fed7aa")]
W = 0.27

for cond, spec in DATA.items():
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    x = 0.0
    ticks, ticklabels = [], []
    def triplet(x0, vals, ref=False):
        for j, ((sname, col), v) in enumerate(zip(STRATA, vals if isinstance(vals, tuple) else [])):
            pass
    for name, vals in REFS:
        for j, ((sname, col), v) in enumerate(zip(STRATA, vals)):
            ax.bar(x + (j - 1) * W, v, W, color=col,
                   edgecolor="#4a5568", lw=0.7)
            ax.text(x + (j - 1) * W, v + 0.4, f"{v:.0f}", ha="center", fontsize=7)
        ax.axvspan(x - 0.55, x + 0.55, color="#000000", alpha=0.04, zorder=0)
        ticks.append(x)
        ticklabels.append("★ " + name)
        x += 1.35
    x += 0.55
    for model, cells in spec["models"].items():
        cxs = []
        for i, cell in enumerate(cells):
            if cell is None:
                ax.bar(x, 52, 3 * W, color="none", edgecolor="#a0aec0", ls="--",
                       lw=1.0, hatch="//", alpha=0.30)
                note = spec["pending"].get((model, i), "not yet run")
                ax.text(x, 17.5, note, ha="center", va="bottom", fontsize=7.5,
                        color="#718096", style="italic", rotation=90)
            else:
                for j, ((sname, col), v) in enumerate(zip(STRATA, cell)):
                    ax.bar(x + (j - 1) * W, v, W, color=col)
                    ax.text(x + (j - 1) * W, v + 0.4, f"{v:.0f}", ha="center", fontsize=7)
            role = spec["roles"][i]
            ax.text(x, 15.4, role.replace(" rules", "\nrules"), ha="center", va="top",
                    fontsize=7.2, color="#4a5568")
            cxs.append(x)
            x += 1.35
        ticks.append(sum(cxs) / len(cxs))
        ticklabels.append("\n\n" + model)
        x += 0.7
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in STRATA]
    ax.legend(handles, [s for s, _ in STRATA], fontsize=8.5, loc="upper right")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=9.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("sealed success % — y-axis starts at 15")
    ax.set_ylim(15, 54)
    ax.grid(axis="y", alpha=0.25)
    title_cond = ("ADVERSARIAL input (repair)" if cond == "adversarial"
                  else "ORIGINAL input (polish) — originals reference = the human phrasing")
    ax.set_title(f"Sealed ladder — {title_cond}    (★ reference | *oracle selected on layouts 0-17)",
                 fontsize=10.5)
    if cond == "adversarial":
        fig.text(0.01, 0.005, "all bare cells use the unified bare template (rules skeleton minus rules) "
                 "+ full Gemini trace; the legacy CoVer-scaffolded Qwen bare row (31.0) is retained only "
                 "in the appendix scoreboard", fontsize=6.6, color="#4a5568")
    fig.tight_layout()
    out = f"results/charts/sealed_ladder_{cond}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.25)
    print("chart ->", out)
