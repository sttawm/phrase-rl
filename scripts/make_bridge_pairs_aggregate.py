#!/usr/bin/env python3
"""results/charts/pairs_aggregate_bridge.png — Bridge single-edit pair search
aggregated by edit category: verified significant vs null, from
pair_verdicts.csv (105 candidate pairs, n=72 verification rolls each).
Classification mirrors the paper's swing categories; rows whose two phrases
differ in more than one concept are counted as multi-edit cross-pairs.
No in-image title (paper style)."""
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
v = pd.read_csv(R / "results/analysis/pairverify/pair_verdicts.csv")
COLORS = {"black","blue","green","yellow","purple","grey","gray","orange","red"}
PREPS = {"on","onto","atop","into","in","inside","to","top","of","upon"}
VERBS = {"put","place","set","stack","move","rest","position","deposit","drop","grab","pick","lay"}


def toks(s):
    return re.findall(r"[a-z']+", s.lower())


def classify(r):
    g = r.group
    if g.startswith(("A01", "A02")):
        return "Phrase structure"
    if g.startswith(("A03", "A04", "A05", "A06", "A07")):
        return "Preposition"
    if g.startswith(("A08", "A09", "A10", "A11", "A12")):
        return "Action verb"
    if g == "A17_noun_dish_plate":
        return "Destination noun synonym"
    if g.startswith(("A13", "A14", "A15", "A16", "Rn")):
        return "Source noun synonym"
    if g in ("A18_order_fronted", "A19_articles", "A20_polite_colororder"):
        return "Phrase structure"
    lo, hi = toks(r.phrase_lo), toks(r.phrase_hi)
    d_lo = [w for w in lo if w not in hi]
    d_hi = [w for w in hi if w not in lo]
    diff = set(d_lo) | set(d_hi)
    if r.phrase_lo.lower().strip(".,") == r.phrase_hi.lower().strip(".,"):
        return "Phrase structure"
    if diff and diff <= COLORS or g.startswith("Ac"):
        return ("Color: addition helps" if set(d_hi) & COLORS
                else "Color: addition hurts")
    if diff and diff <= PREPS:
        return "Preposition"
    if diff and diff <= VERBS:
        return "Action verb"
    if diff and diff <= {"the", "a"}:
        return "Phrase structure"
    if diff and len(d_lo) <= 2 and len(d_hi) <= 2 and not diff & (VERBS | PREPS):
        return ("Destination noun synonym"
                if (lo and (lo[-1] in diff or (len(lo) > 1 and lo[-2] in diff)))
                else "Source noun synonym")
    return "Multi-edit cross-pairs"


v["cat"] = v.apply(classify, axis=1)
ORDER = ["Source noun synonym", "Destination noun synonym", "Preposition",
         "Color: addition helps", "Color: addition hurts", "Action verb",
         "Phrase structure", "Multi-edit cross-pairs"]
agg = v.groupby("cat").agg(sig=("survives", "sum"), tested=("survives", "size"))
agg = agg.reindex(ORDER)
agg["null"] = agg.tested - agg.sig

C_SIG, C_NULL, INK = "#3F6B52", "#D9DEE6", "#2d3748"
fig, ax = plt.subplots(figsize=(5.6, 2.9))
y = range(len(agg))[::-1]
ax.barh(y, agg.sig, 0.62, color=C_SIG, edgecolor="none", label="significant")
ax.barh(y, agg.null, 0.62, left=agg.sig, color=C_NULL, edgecolor="#6E7B8B",
        lw=0.6, label="null")
for yi, (_, r) in zip(y, agg.iterrows()):
    if r.sig:
        ax.text(r.sig / 2, yi, str(int(r.sig)), ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")
    if r.null:
        ax.text(r.sig + r.null / 2, yi, str(int(r.null)), ha="center",
                va="center", fontsize=8, color=INK)
    ax.text(r.tested + 0.5, yi, f"{100*r.sig/r.tested:.0f} %", va="center",
            fontsize=7.5, color="#4a5568")
ax.set_yticks(list(y))
ax.set_yticklabels(agg.index, fontsize=8.5)
ax.set_xlabel("candidate pairs verified at n = 72 (significant / null)",
              fontsize=8)
ax.set_xlim(0, agg.tested.max() * 1.14)
ax.legend(fontsize=7.5, loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.15)
fig.tight_layout()
out = R / "results/charts/pairs_aggregate_bridge.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.12)
print("chart ->", out)
print(agg.to_string())
