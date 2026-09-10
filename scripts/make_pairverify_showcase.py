#!/usr/bin/env python3
"""results/charts/pairverify_showcase.png — minimal-pair verification showcase.

Six panels in the spread-showcase form: task first-frame at left, horizontal
success bars at right, one bar per phrase, the changed tokens in bold, Wilson
95% whiskers. All values from the full-grid re-roll (24 layouts x 3 reps,
n=72/phrase; episodes_*.parquet under results/analysis/pairverify/).
"""
import re

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

S = (pd.read_csv("results/analysis/pairverify/phrase_summary.csv")
     .drop_duplicates(subset=["task", "phrase"]))
FR = {"coke_can_on_ramekin": "/tmp/extra_frames/widowx_coke_can_on_ramekin_clean.png",
      "pepsi_on_plate": "/tmp/sealed12_frames/widowx_pepsi_on_plate_clean.png",
      "eggplant_on_sponge": "/tmp/sealed12_frames/widowx_eggplant_on_sponge_clean.png",
      "eggplant_on_keyboard": "/tmp/sealed12_frames/widowx_eggplant_on_keyboard_clean.png",
      "cube_on_plate": "/tmp/sealed12_frames/widowx_cube_on_plate_clean.png",
      "carrot_on_ramekin": "/tmp/sealed12_frames/widowx_carrot_on_ramekin_clean.png"}

# (panel title, task-short, [(phrase, [tokens to bold])]) — phrases exact
PANELS = [
 ("Destination-noun ladder — one word decides (coke can → white vessel)", "coke_can_on_ramekin", [
   ("place the red cola can inside the white container", ["container"]),
   ("place the red cola can inside the white basin", ["basin"]),
   ("place the red cola can inside the white bowl", ["bowl"]),
   ("place the red cola can inside the white pot", ["pot"]),
   ("place the red cola can inside the white dish", ["dish"]),
   ("place the red cola can inside the white cup", ["cup"]),
   ("place the red cola can inside the white ramekin", ["ramekin"]),
 ]),
 ("Capitalisation and a full stop", "pepsi_on_plate", [
   ("set the pepsi down on the plate", []),
   ("Set the Pepsi down on the plate.", ["Set", "Pepsi", "."]),
   ("put the pepsi on the plate", []),
   ("Put the Pepsi on the plate.", ["Put", "Pepsi", "."]),
 ]),
 ("Word order only", "eggplant_on_sponge", [
   ("Set the purple eggplant on the sponge.", []),
   ("purple Eggplant set on the sponge.", ["purple", "Eggplant", "set"]),
 ]),
 ("Adding the destination's colour", "eggplant_on_keyboard", [
   ("put the eggplant on the black keyboard", ["black"]),
   ("put the eggplant on the keyboard", []),
 ]),
 ("Verb synonym: put → place", "cube_on_plate", [
   ("place the green cube into the dish", ["place"]),
   ("put the green cube into the dish", ["put"]),
 ]),
 ("Noun hypernym: carrot → vegetable", "carrot_on_ramekin", [
   ("Pick up the orange carrot and place it in the white bowl.", ["carrot"]),
   ("Pick up the orange vegetable and place it in the white bowl.", ["vegetable"]),
 ]),
]

C_HI, C_LO = "#a3bffa", "#feb2b2"


def fmt(phrase, bold):
    """Phrase with chosen tokens in mathtext bold (keeps everything else plain)."""
    out = []
    for tok in phrase.split(" "):
        core = tok
        boldit = any(b != "." and core.strip(".,?!").lower() == b.strip(".").lower()
                     and (b[0].isupper() == core[0].isupper() if b[0].isalpha() and core else True)
                     for b in bold if b != ".")
        if boldit:
            m = re.match(r"^(\w+)([.,?!]*)$", core)
            if m:
                core = r"$\bf{" + m.group(1) + r"}$" + m.group(2)
        out.append(core)
    s = " ".join(out)
    if "." in bold and s.endswith("."):
        s = s[:-1] + r"$\bf{.}$"
    return s


nbars = [len(p[2]) for p in PANELS]
fig = plt.figure(figsize=(11.5, sum(nbars) * 0.52 + len(PANELS) * 0.75))
gs = fig.add_gridspec(len(PANELS), 2, width_ratios=[1, 3.6],
                      height_ratios=[n + 1.2 for n in nbars], hspace=0.55, wspace=0.06,
                      top=0.965, bottom=0.035, left=0.03, right=0.97)

for i, (title, short, phrases) in enumerate(PANELS):
    task_rows = S[S.task.str.contains(short)]
    axi = fig.add_subplot(gs[i, 0])
    axi.imshow(mpimg.imread(FR[short]))
    axi.axis("off")
    ax = fig.add_subplot(gs[i, 1])
    vals = []
    for phrase, bold in phrases:
        r = task_rows[task_rows.phrase == phrase]
        assert len(r) == 1, f"missing/dup: {phrase!r}"
        r = r.iloc[0]
        vals.append((phrase, bold, r.succ, r.lo95, r.hi95, int(r.n)))
    ys = range(len(vals))[::-1]
    for y, (phrase, bold, v, lo, hi, n) in zip(ys, vals):
        top = v == max(x[2] for x in vals)
        ax.barh(y, v, 0.6, color=C_HI if top else C_LO, edgecolor="#4a5568", lw=0.7)
        ax.errorbar([v], [y - 0.27], xerr=[[v - lo], [hi - v]], fmt="none",
                    ecolor="#718096", elinewidth=0.9, capsize=2.0)
        ax.text(1.2, y + 0.06, fmt(phrase, bold), va="center", fontsize=9.2, color="#1a202c")
        ax.text(101, y, f"{v:.0f}%", va="center", ha="left", fontsize=10, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(vals) - 0.3)
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(title, fontsize=11, loc="left", pad=6)
    if i == len(PANELS) - 1:
        ax.set_xlabel("rollout success % (24 layouts × 3 reps, n=72 per phrase; Wilson 95%)",
                      fontsize=9)

fig.suptitle("Tiny benign edits, verified at full layout grids — frozen π₀, SIMPLER Bridge",
             fontsize=13, y=0.993)
out = "results/charts/pairverify_showcase.png"
fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.3)
print("chart ->", out)
