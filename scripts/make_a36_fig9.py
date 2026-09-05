#!/usr/bin/env python3
"""A36 replicate results in the paper's Figure-9 layout.

  .venv/bin/python scripts/make_a36_fig9.py <round>   # round = 2 or 3

Identical layout to make_results_translating_v2.py (Baselines block + three
condition panels x three appliers x three rule roles, triplet bars carrying
pooled / in-distribution(5) / out-of-distribution(7), dotted baseline levels).

Data: results/analysis/a36_cells.json (BASE-WEIGHTED, the confirmed A31
convention). Roles map to evidence diets:
  rollout+training-data rules = combined book  (tag b<round>)
  rollout-derived rules       = rollout-only    (tag s<round>)
  no rules (bare prompt)      = scaffold control (A34 natural / A31 adversarial)
Cells with no data yet are drawn hollow and labelled "rolling".
"""
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RND = sys.argv[1] if len(sys.argv) > 1 else "2"
R = pathlib.Path(__file__).resolve().parents[1]
cells = json.loads((R / "results/analysis/a36_cells.json").read_text())

# --- baselines (no rephraser), pooled/iv/oov, from the paper's Fig-9 v2 ---
BASE = [("oracle*", 48.2, 50.3, 46.8, "#2f855a"),
        ("original", 36.1, 48.3, 27.3, "#4a5568"),
        ("natural", 28.45, 41.5, 19.2, "#b7791f"),
        ("adversarial", 24.5, 25.2, 23.9, "#c53030"),
        ("natural,\nstd. $\\pi_0$", 23.8, 35.6, 15.4, "#805ad5")]

# scaffold (no-rules) controls -- pooled/iv/oov per applier per condition.
# natural from A34 (same 186 image set), adversarial from A31 (same 72 attacks).
# original scaffold was never run (X'd out), so that role is empty for Original.
_r1 = json.loads((R / "results/analysis/r1_cells.json").read_text())
def _sc(ap, c):
    r = _r1.get(f"sc|{ap}|{c}")
    return (r["pooled"], r.get("iv"), r.get("oov")) if r else None
SCAFFOLD = {"Adversarial": {ap: _sc(ap, "adv") for ap in ("gemini", "claude", "qwen")},
            "Natural": {ap: _sc(ap, "nat") for ap in ("gemini", "claude", "qwen")},
            "Original": {}}
ROLES = ["rollout+\ntraining-data\nrules", "rollout-\nderived\nrules",
         "no rules\n(bare\nprompt)"]
MODELS = [("Gemini-Pro", "gemini"), ("Claude Fable", "claude"), ("Frozen Qwen", "qwen")]
COND2C = {"Adversarial": "adv", "Natural": "nat", "Original": "orig"}


def cell(role_tag, applier, cond):
    """role_tag in {b,s}; returns (pooled,iv,oov) or None."""
    k = f"{role_tag}{RND}|{applier}|{COND2C[cond]}"
    r = cells.get(k)
    if not r:
        return None
    return (r["pooled"], r.get("iv"), r.get("oov"))


C_RULES, C_BASEBAR = "#a3bffa", "#8b96a5"
C_IV, C_OOV = "#b2f5ea", "#fed7aa"
YMIN, YMAX = 15, 52

fig, axes = plt.subplots(1, 4, figsize=(21.8, 5.8), sharey=True,
                         gridspec_kw={"width_ratios": [1.75, 3, 3, 2.15]})


def triplet(ax, x, cell, pooled_color):
    if cell is None:
        ax.bar(x, YMAX - YMIN - 20, 0.42, bottom=YMIN, color="none",
               edgecolor="#cbd5e0", lw=0.9, ls=":", zorder=2)
        ax.text(x, YMIN + 6, "rolling", ha="center", va="center", fontsize=6.2,
                color="#a0aec0", rotation=90)
        return
    p, iv, oo = cell
    if iv is not None:
        ax.bar(x - 0.20, iv, 0.34, color=C_IV, alpha=0.55, zorder=2)
        ax.text(x - 0.32, iv + 0.4, f"{iv:.0f}", ha="center", fontsize=6.6,
                color="#4a5568", zorder=4)
    if oo is not None:
        ax.bar(x + 0.20, oo, 0.34, color=C_OOV, alpha=0.55, zorder=2)
        ax.text(x + 0.32, oo + 0.4, f"{oo:.0f}", ha="center", fontsize=6.6,
                color="#4a5568", zorder=4)
    ax.bar(x, p, 0.42, color=pooled_color, zorder=3, edgecolor="#4a5568", lw=0.9)
    ax.text(x, p + 0.55, f"{p:.1f}", ha="center", fontsize=8.6,
            fontweight="bold", zorder=4)


axB = axes[0]
for i, (name, p, iv, oo, c) in enumerate(BASE):
    triplet(axB, float(i), (p, iv, oo), C_BASEBAR)
    axB.text(i, YMIN - 1.1, name.replace("adversarial", "adver-\nsarial").replace("oracle*", "oracle$^{*}$"),
             ha="center", va="top", fontsize=7.2, color="#4a5568")
axB.set_title("Baselines\n(no rephraser)", fontsize=11.5, pad=10)
axB.set_xlim(-0.7, len(BASE) - 0.3)
axB.set_xticks([])
axB.set_ylabel("sealed-suite success %", fontsize=11)

for ax, cond in zip(axes[1:], ["Adversarial", "Natural", "Original"]):
    x = 0.0
    ticks, tlabels = [], []
    for mi, (model, ap) in enumerate(MODELS):
        cxs = []
        roles = [("b", C_RULES), ("s", C_RULES)]
        if cond != "Original":
            roles.append(("sc", C_BASEBAR))
        for ri, (rt, col) in enumerate(roles):
            c = SCAFFOLD[cond].get(ap) if rt == "sc" else cell(rt, ap, cond)
            triplet(ax, x, c, col)
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
    ax.set_title("Translating $\\bf{%s}$ Task Instructions" % cond, fontsize=12.5, pad=10)

for ax in axes:
    for name, p, iv, oo, c in BASE:
        ax.axhline(p, color=c, lw=1.2, ls=(0, (3, 3)), zorder=1, alpha=0.85)
axes[0].set_ylim(YMIN, YMAX)

axR = axes[3]
for name, p, iv, oo, c in BASE:
    axR.text(axR.get_xlim()[1] + 0.06, p, name.rstrip("*").replace(",\n", " "),
             fontsize=8.2, color=c, va="center", ha="left", clip_on=False)

handles = [plt.Rectangle((0, 0), 1, 1, color=C_RULES),
           plt.Rectangle((0, 0), 1, 1, color=C_BASEBAR),
           plt.Rectangle((0, 0), 1, 1, color=C_IV, alpha=0.55),
           plt.Rectangle((0, 0), 1, 1, color=C_OOV, alpha=0.55),
           plt.Line2D([0], [0], color="#4a5568", lw=1.2, ls=(0, (3, 3)))]
axes[1].legend(handles, ["rules cell (pooled)", "baseline (pooled)",
                         "in-distribution (5)", "out-of-distribution (7)", "baseline level"],
               fontsize=7.4, loc="upper left", framealpha=0.95)

fig.suptitle(f"A36 replicate round r{RND} — base-weighted, sealed 12 tasks "
             "(natural=186 phrases; scaffold = no-rules control)",
             fontsize=11, y=1.02)
fig.tight_layout(rect=(0, 0, 0.985, 1.0))
out = R / f"results/charts/a36_fig9_r{RND}.png"
fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.25)
print("chart ->", out)
