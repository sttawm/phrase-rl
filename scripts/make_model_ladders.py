#!/usr/bin/env python3
"""Per-model sealed ladders: one figure per rewriter model, two panels
(adversarial | original). Bars: references (oracle / original / adversarial
passthrough) then Model bare -> Model + Rules+rollout (v3) -> Model +
Rules-train+rollout (v4). Missing cells render as hatched placeholders.
Values = pooled sealed (12 tasks x 24 x 12)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = {"oracle": 48.23, "original": 36.08, "adversarial (pass through)": 26.59}

MODELS = {
    "qwen":   ("Frozen Qwen (9B, Gemini scene trace)", {
        "adv": [("Qwen frozen rephraser", 30.96), ("Qwen + Rules+rollout (v3)", 31.48),
                ("Qwen + Rules-train+rollout (v4)", 30.06)],
        "nom": [("Qwen frozen rephraser", None), ("Qwen + Rules+rollout (v3)", None),
                ("Qwen + Rules-train+rollout (v4)", 27.81)],
    }),
    "gemini": ("Gemini-Pro (with reasoning)", {
        "adv": [("Gemini bare", 27.78), ("Gemini + Rules+rollout (v3)", 31.57),
                ("Gemini + Rules-train+rollout (v4)", 33.30)],
        "nom": [("Gemini bare", None), ("Gemini + Rules+rollout (v3)", 34.78),
                ("Gemini + Rules-train+rollout (v4)", 37.18)],
    }),
    "claude": ("Claude Fable 5", {
        "adv": [("Claude bare", None), ("Claude + Rules+rollout (v3)", 30.99),
                ("Claude + Rules-train+rollout (v4)", 34.26)],
        "nom": [("Claude bare", None), ("Claude + Rules+rollout (v3)", None),
                ("Claude + Rules-train+rollout (v4)", 37.73)],
    }),
}
PENDING = {("claude", "adv", "Claude bare"): "rolling now (arm H)"}


def panel(ax, refs, bars, title):
    rows = [(n, v, True) for n, v in refs] + [(n, v, False) for n, v in bars]
    rows = rows[::-1]  # top-down reading order
    ys = range(len(rows))
    for y, (name, val, is_ref) in zip(ys, rows):
        if val is None:
            ax.barh(y, 50, 0.62, color="none", edgecolor="#a0aec0", ls="--", lw=1.0, hatch="//", alpha=0.35)
            ax.text(1.0, y, "not yet run", va="center", fontsize=7.5, color="#718096", style="italic")
        else:
            color = "#cbd5e0" if is_ref else "#a3bffa"
            ax.barh(y, val, 0.62, color=color, edgecolor="#4a5568" if is_ref else "none", lw=0.8)
            ax.text(val + 0.5, y, f"{val:.1f}", va="center", fontsize=8.5,
                    fontweight="bold" if is_ref else "normal")
        if is_ref:
            ax.axhspan(y - 0.42, y + 0.42, color="#000000", alpha=0.04, zorder=0)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([("★ " if r else "") + n for n, _, r in rows], fontsize=9)
    ax.set_xlim(0, 55)
    ax.set_xlabel("sealed success % (12 tasks × 24 × 12)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.25)


FOOT = ("★ reference   |   v3 = Rules+rollout (rollout-derived)   |   "
        "v4 = Rules-train+rollout (train-mined + rollout)   |   *oracle selected on layouts 0-17")
for key, (label, data) in MODELS.items():
    adv_bars = [(n, v) for n, v in data["adv"]]
    for i, (n, v) in enumerate(adv_bars):
        if (key, "adv", n) in PENDING and v is None:
            adv_bars[i] = (n + f"  ({PENDING[(key, 'adv', n)]})", None)
    for cond, refs, bars, cname in [
            ("adversarial", [("oracle*", REF["oracle"]), ("original", REF["original"]),
                             ("adversarial (pass through)", REF["adversarial (pass through)"])],
             adv_bars, "ADVERSARIAL input (repair)"),
            ("original", [("oracle*", REF["oracle"]), ("original", REF["original"])],
             data["nom"], "ORIGINAL input (polish)")]:
        fig, ax = plt.subplots(figsize=(9.5, 3.4))
        panel(ax, refs, bars, f"{label} — {cname}")
        fig.text(0.02, 0.005, FOOT, fontsize=6.6, color="#4a5568")
        fig.tight_layout()
        out = f"results/charts/model_ladder_{key}_{cond}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.28)
        print("chart ->", out)
