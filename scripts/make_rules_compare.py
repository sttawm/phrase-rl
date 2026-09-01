#!/usr/bin/env python3
"""results/charts/rules_compare_<run>.png -- all passes on one chart.

Three panels (train / val_held / val8), x = iteration, one line per rephraser,
y = delta in mean proxy logit vs the unrephrased bases. Partial iterations
(train scored, validation pending) contribute their train point only.

  .venv/bin/python scripts/make_rules_compare.py r1
"""
import argparse
import glob
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
COLORS = {"gemini": "#2563eb", "claude": "#d97706", "qwen": "#059669"}
PANELS = [("train", "train (96 bases)"), ("val_held", "val_held (held-out tasks)"),
          ("val8", "val8 (classic 8)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args()
    run = ROOT / "results/rules_runs" / args.run_id

    series = {}   # pass -> metric -> [(iter, delta)]
    for pdir in sorted(run.glob("pass_*")):
        name = pdir.name.split("_", 1)[1]
        s = {m: [] for m, _ in PANELS}
        for f in sorted(pdir.glob("iter_*/scores.json")):
            it = int(f.parent.name.split("_")[1])
            d = json.load(open(f))["delta"]
            for m, _ in PANELS:
                if d.get(m) is not None:
                    s[m].append((it, d[m]))
        # partial iterations: train-only points
        for f in sorted(pdir.glob("iter_*/eval_train.json")):
            it = int(f.parent.name.split("_")[1])
            if not (f.parent / "scores.json").exists():
                s["train"].append((it, json.load(open(f))["delta"]))
        series[name] = s

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True)
    for ax, (m, title) in zip(axes, PANELS):
        for name, s in series.items():
            pts = sorted(s[m])
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                    color=COLORS.get(name, "#6b7280"), label=name, lw=2, ms=5)
        ax.axhline(0, color="#6b7280", lw=1, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0].set_ylabel("delta vs unrephrased bases (mean logit)")
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(f"rules loop {args.run_id} -- all passes "
                 f"(iteration 0 = empty rulebook)", fontsize=11)
    fig.tight_layout()
    out = ROOT / "results/charts" / f"rules_compare_{args.run_id}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"chart -> {out}")


if __name__ == "__main__":
    main()
