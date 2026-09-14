#!/usr/bin/env python3
"""results/charts/swings_lollipop.png — the collaborator's dumbbell idiom
applied to the CURRENT curated swing entries (parsed from the paper's two
fragment files): one row per entry, dots = phrase success rates (filled =
best), line in category color, swing pp at right, sorted by swing."""
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
CAT_COLOR = {"noun": "#4C72B0", "color": "#DD8452", "verb": "#55A868",
             "prep": "#B8962E", "form": "#8172B3"}
CAT_OF = {"Source noun synonym": ("source noun", "noun"),
          "Destination noun synonym": ("destination noun", "noun"),
          "Preposition": ("preposition", "prep"),
          "Color --- addition helps": ("color", "color"),
          "Color --- addition hurts": ("color", "color"),
          "Action verb": ("verb", "verb"),
          "Phrase structure": ("structure", "form")}


def parse(fname):
    t = (R / "paper/figures" / fname).read_text()
    entries, cat = [], None
    for line in t.splitlines():
        m = re.match(r"\{\\bfseries (.+)\}\\par\\nopagebreak", line)
        if m and m.group(1) in CAT_OF:
            cat = m.group(1)
            continue
        m = re.match(r"\{\\bfseries ([a-z\\_/0-9]+)\}(?:\\enspace\{\\footnotesize\(([a-z ]+)\)\})?\\enspace \+(\d+)\\,pp\\,\$\\cdot\$\\, (p[=<][0-9.]+)\\par", line)
        if m:
            task = m.group(1).replace("\\_", "_")
            sub = m.group(2)
            label, ck = CAT_OF[cat]
            if sub:
                label = sub
            entries.append(dict(task=task, label=label, ck=ck,
                                gap=int(m.group(3)), vals=[], best=None))
            continue
        m = re.match(r"\\hangindent=[0-9.]+em \\pct\{(\w+)\}\{(\d+)\}~", line)
        if m and entries:
            v = int(m.group(2))
            entries[-1]["vals"].append(v)
            if m.group(1) == "hipct":
                entries[-1]["best"] = v
    return entries


def panel(ax, entries, title):
    entries = sorted(entries, key=lambda e: -e["gap"])
    for i, e in enumerate(entries):
        y = len(entries) - 1 - i
        c = CAT_COLOR[e["ck"]]
        ax.plot([min(e["vals"]), max(e["vals"])], [y, y], color=c, lw=2.2,
                alpha=0.55, zorder=1, solid_capstyle="round")
        for v in e["vals"]:
            if v == e["best"]:
                ax.plot(v, y, "o", ms=5.5, color=c, zorder=3)
            else:
                ax.plot(v, y, "o", ms=4.6, mfc="white", mec=c, mew=1.2, zorder=2)
        ax.text(104, y, f"+{e['gap']}", va="center", fontsize=7.5,
                color="#4a5568")
    ax.set_yticks(range(len(entries))[::-1])
    ax.set_yticklabels([f"{e['task']} · {e['label']}" for e in entries],
                       fontsize=7.5)
    ax.set_xlim(-2, 113)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=8)
    ax.set_ylim(-0.7, len(entries) - 0.3)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=10)
    ax.text(104, len(entries) - 0.1, "swing", fontsize=7.5, color="#4a5568")


B = parse("pairverify_bridge_body.tex")
L = parse("pairverify_libero_body.tex")
print(f"parsed: bridge {len(B)}, libero {len(L)}")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 0.185 * (len(B) + len(L)) + 1.8),
                             gridspec_kw={"height_ratios": [len(B), len(L)]})
panel(a1, B, "$\\pi_0$ on SIMPLER Bridge (n = 72 per phrase)")
panel(a2, L, "$\\pi_{0.5}$ on LIBERO (n = 50 per phrase)")
handles = [plt.Line2D([0], [0], marker="o", ls="", color=CAT_COLOR[k], ms=6)
           for k in ["noun", "color", "verb", "prep", "form"]]
handles += [plt.Line2D([0], [0], marker="o", ls="", color="#555", ms=6),
            plt.Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555", ms=5.5)]
fig.legend(handles, ["Noun", "Color or modifier", "Verb",
                     "Preposition or particle", "Form, case, order",
                     "best phrase", "other phrases"],
           fontsize=7.5, ncol=4, loc="lower center", frameon=False,
           bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.045, 1, 1))
out = R / "results/charts/swings_lollipop.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
print("chart ->", out)
