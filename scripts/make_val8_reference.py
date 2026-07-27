#!/usr/bin/env python3
"""Val-8 reference chart (provisional, banked board data): Original phrasing vs
Oracle (search-best) per stratum on the 8 development tasks — the comparison
frame for RL rollout numbers. Mixed n (36-180) noted; the rolled reference leg
(Orig+Adv+Oracle x 24x12) replaces this when it lands.
"""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NOM = {"widowx_carrot_on_plate": "put carrot on plate",
       "widowx_spoon_on_towel": "put the spoon on the towel",
       "widowx_stack_cube": "stack the green block on the yellow block",
       "widowx_put_eggplant_in_basket": "put eggplant into yellow basket",
       "widowx_carrot_on_keyboard": "put carrot on keyboard",
       "widowx_carrot_on_wheel": "put carrot on wheel",
       "widowx_coke_can_on_ramekin": "put coke can on ramekin",
       "widowx_coke_can_on_plate": "put coke can on plate"}
IV = ["widowx_carrot_on_plate", "widowx_spoon_on_towel", "widowx_stack_cube",
      "widowx_put_eggplant_in_basket"]
OO = [t for t in NOM if t not in IV]
KEYMAP = {"spoon": "widowx_spoon_on_towel", "stack": "widowx_stack_cube",
          "carrotplate": "widowx_carrot_on_plate", "eggplant": "widowx_put_eggplant_in_basket",
          "keyboard": "widowx_carrot_on_keyboard", "wheel": "widowx_carrot_on_wheel",
          "ramekin": "widowx_coke_can_on_ramekin", "cokeplate": "widowx_coke_can_on_plate"}

boards = {}
for f in sorted(glob.glob("results/search/*_results.json")):
    base = f.split("/")[-1]
    if base.startswith("sealedsearch"):
        continue
    key = next((k for k in KEYMAP if base.startswith(k)), None)
    if key is None:
        continue
    t = KEYMAP[key]
    for e in json.load(open(f)).get("scoreboard", []):
        p = e["phrase"].strip()
        boards.setdefault(t, {})
        if p not in boards[t] or e.get("n", 0) > boards[t][p][1]:
            boards[t][p] = (e["success_pct"], e.get("n", 0))

orig, orac = {}, {}
for t, ph in boards.items():
    nom = NOM[t].strip()
    hit = next(((s, n) for p, (s, n) in ph.items() if p.lower() == nom.lower()), None)
    orig[t] = hit
    cands = [(s, n) for _, (s, n) in ph.items() if n >= 36]
    orac[t] = max(cands) if cands else None
    print(f"{t.replace('widowx_',''):26s} nominal {hit} | oracle {max(cands) if cands else None}")

def strata(d):
    vals = {k: v[0] for k, v in d.items() if v}
    return (sum(vals.values()) / len(vals),
            sum(vals[t] for t in IV if t in vals) / len([t for t in IV if t in vals]),
            sum(vals[t] for t in OO if t in vals) / len([t for t in OO if t in vals]))

rows = [("Oracle (search-best)", strata(orac), "#822727"),
        ("Orig", strata(orig), "#48bb78")]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
y = [1, 0]
h = 0.24
for yi, (label, (pool, iv, oo), color) in zip(y, rows):
    ax.barh(yi + h, pool, h, color=color)
    ax.barh(yi, iv, h, color=color, alpha=0.55)
    ax.barh(yi - h, oo, h, color=color, alpha=0.3, hatch="//", edgecolor=color, lw=0)
    ax.text(pool + 0.5, yi + h, f"{pool:.1f}", va="center", fontsize=11, fontweight="bold")
    ax.text(iv + 0.5, yi, f"{iv:.1f}", va="center", fontsize=8, color="#4a5568")
    ax.text(oo + 0.5, yi - h, f"{oo:.1f}", va="center", fontsize=8, color="#4a5568")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=12)
ax.legend(handles=[Patch(color="#4a5568", label="pooled (8 tasks)"),
                   Patch(color="#4a5568", alpha=0.55, label="in-vocab (4 tasks)"),
                   Patch(facecolor="#4a5568", alpha=0.3, hatch="//", label="out-of-vocab (4 tasks)")],
          loc="lower right", fontsize=8)
ax.set_xlabel("success % — 8 development tasks (banked search data, n=36-180 per phrase)")
ax.set_title("Reference frame on the 8 development tasks (provisional — rolled Adv/Orig/Oracle leg pending)")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig("results/charts/val8_reference.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
print("chart -> results/charts/val8_reference.png")
