#!/usr/bin/env python3
"""In-vocab vs out-of-vocab decomposition of every checkpoint eval (val-8, tagged).

Two panels (polish | repair). Per panel: v7a curve 0-340, v7f curve plotted as a
branch segment at x = 160 + step (v7f forked from v7e@20 which forked from
v7a@140), IV solid / OOV dashed, references from val8_reference_x12 (12-rep).
Strata: IV = carrot_on_plate, spoon_on_towel, stack_cube, put_eggplant_in_basket
(all nouns audit-nonzero); OOV = keyboard/wheel/ramekin/coke tasks (each carries
an audit-confirmed zero-count noun). Equal task weighting (n equal per task).
"""
import glob
import json

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IV = {"widowx_carrot_on_plate", "widowx_spoon_on_towel", "widowx_stack_cube",
      "widowx_put_eggplant_in_basket"}


def merged(*patterns):
    by_step = {}
    for pat in patterns:
        for f in glob.glob(pat):
            for l in open(f):
                l = l.strip()
                if not l or l.startswith(("<", "=", ">")):
                    continue
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                s = r["step"]
                if s not in by_step or r.get("n", 0) > by_step[s].get("n", 0):
                    by_step[s] = r
    return dict(sorted(by_step.items()))


def strata(r):
    pt = r["per_task"]
    iv = [v for t, v in pt.items() if t in IV]
    oo = [v for t, v in pt.items() if t not in IV]
    return sum(iv) / len(iv), sum(oo) / len(oo)


series = {
    ("v7a", "pol"): merged("results/analysis/v7_dev8_tagged*.jsonl"),
    ("v7a", "rep"): merged("results/analysis/v7_dev8adv_tagged*.jsonl"),
    ("v7f", "pol"): merged("results/analysis/v7f_pol_curve.jsonl",
                           "results/analysis/_scp_v7f_pol_curve*.jsonl"),
    ("v7f", "rep"): merged("results/analysis/v7f_adv_curve.jsonl",
                           "results/analysis/v7f_adv_curve_rep1.jsonl"),
}
ref = pd.read_parquet("results/sealed/val8_reference_x12.parquet")
R = {}
for arm in ("val8_orig", "val8_adv", "val8_oracle"):
    x = ref[ref.arm == arm]
    pt = {t: float(g.success.mean() * 100) for t, g in x.groupby("task")}
    R[arm] = (sum(v for t, v in pt.items() if t in IV) / 4,
              sum(v for t, v in pt.items() if t not in IV) / 4)

FOFF = 160  # v7f branch offset: v7a@140 -> v7e 20 steps -> v7f step 0

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
for ax, cond, title, passthru in [
        (axes[0], "pol", "POLISH — rewriting original instructions", "val8_orig"),
        (axes[1], "rep", "REPAIR — rewriting adversarial instructions", "val8_adv")]:
    piv, poo = R[passthru]
    ax.axhline(piv, color="#2f855a", ls="--", lw=1.4)
    ax.axhline(poo, color="#2f855a", ls=":", lw=1.4)
    ax.text(342, piv + 0.5, f"no-rewrite IV {piv:.1f}", color="#2f855a", fontsize=7, ha="right")
    ax.text(342, poo + 0.5, f"no-rewrite OOV {poo:.1f}", color="#2f855a", fontsize=7, ha="right")
    oiv, ooo = R["val8_oracle"]
    ax.axhline(oiv, color="#822727", ls="--", lw=1.0, alpha=0.5)
    ax.axhline(ooo, color="#822727", ls=":", lw=1.0, alpha=0.5)
    ax.text(342, oiv + 0.5, f"oracle IV {oiv:.1f}", color="#822727", fontsize=7, alpha=0.8, ha="right")
    ax.text(342, ooo + 0.5, f"oracle OOV {ooo:.1f}", color="#822727", fontsize=7, alpha=0.8, ha="right")

    for (model, c), style in [(("v7a", cond), dict(color="#6b46c1")),
                              (("v7f", cond), dict(color="#2b6cb0"))]:
        ss = series[(model, c)]
        if not ss:
            continue
        xs = [s + (FOFF if model == "v7f" else 0) for s in ss]
        ivs, oos = zip(*[strata(r) for r in ss.values()])
        deep = [r.get("n", 0) >= 384 for r in ss.values()]
        ax.plot(xs, ivs, marker="o", ms=5, lw=1.8, ls="-",
                label=f"{model} IN-VOCAB", **style)
        ax.plot(xs, oos, marker="o", ms=5, lw=1.5, ls="--", markerfacecolor="none",
                label=f"{model} OUT-OF-VOCAB", **style)
        for x_, y_, d in zip(xs, list(ivs) + list(oos), deep + deep):
            if d:
                ax.plot([x_], [y_], marker="o", ms=7, mfc="none", mec="black", mew=0.7, ls="")
    ax.axvline(FOFF, color="#a0aec0", ls=":", lw=1)
    ax.text(FOFF + 2, 12, "v7f fork (ancestry v7a@140 + v7e 20)", rotation=90,
            fontsize=6.5, color="#718096", va="bottom")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("v7a step  (v7f branch plotted at 160 + v7f step)")
    ax.grid(alpha=0.25)
    ax.set_ylim(10, 64)
axes[0].set_ylabel("val-8 rollout success % (stratum mean of 4 tasks)")
axes[0].legend(fontsize=7.5, loc="lower right")
fig.suptitle("Where the rewriters earn their keep: in-vocab vs out-of-vocab strata, every tagged checkpoint eval\n"
             "(black-edged = 2-rep n=384; others rep-1 n=192, ~-3pp seedset bias; strata are 4 tasks each, SE ~3.5-5pp)",
             fontsize=9.5)
fig.tight_layout()
fig.savefig("results/charts/v7_strata.png", dpi=140, bbox_inches="tight")
print("chart -> results/charts/v7_strata.png")
