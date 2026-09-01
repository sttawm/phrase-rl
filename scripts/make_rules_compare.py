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
          ("val8", "val8 (classic 8)"),
          ("val8r", "val8 recomposed (31 nat / 13 adv / 10 unk)")]


def val8_recomposed_subset(run):
    """Fixed (task, base) subset: all natural + all adversarial + a seeded 10
    of the legacy 'unknown' phrases. The full val8 draw is 52 unknown -- mostly
    search-era phrases with no rewrite headroom -- which drowns the regimes we
    actually care about."""
    import pandas as pd
    bank = pd.read_parquet(run / "bank.parquet")
    kinds = bank.drop_duplicates(["task", "phrase"])[["task", "phrase", "kind"]]
    ev = None
    for f in sorted(run.glob("pass_*/iter_*/eval_val8_*.parquet")):
        ev = pd.read_parquet(f, columns=["task", "base"]); break
    if ev is None:
        return None
    m = ev.drop_duplicates().rename(columns={"base": "phrase"}).merge(
        kinds, on=["task", "phrase"], how="left")
    m["kind"] = m.kind.fillna("unknown")
    keep = m[m.kind.isin(["natural", "adversarial"])]
    unk = m[~m.kind.isin(["natural", "adversarial"])].sample(
        n=min(10, (~m.kind.isin(["natural", "adversarial"])).sum()), random_state=17)
    sub = pd.concat([keep, unk])
    return set(zip(sub.task, sub.phrase))


def val8_recomposed_series(run):
    """pass -> [(iter, mean delta on the recomposed subset)]."""
    import pandas as pd
    sub = val8_recomposed_subset(run)
    if not sub:
        return {}
    rows = list(run.glob("baseline_val8_*_rows.parquet"))
    if not rows:
        return {}
    b = pd.read_parquet(rows[0]).dropna(subset=["z", "grip"])
    b["lg"] = 8.123 + 0.4445 * b.z.astype(float) + 11.3193 * (-b.grip.astype(float))
    base_lg = {(t, p): v for t, p, v in zip(b.task, b.phrase, b.lg)}
    out = {}
    for pdir in sorted(run.glob("pass_*")):
        name = pdir.name.split("_", 1)[1]
        pts = []
        for f in sorted(pdir.glob("iter_*/eval_val8_*.parquet")):
            it = int(f.parent.parts[-1].split("_")[1])
            e = pd.read_parquet(f).dropna(subset=["z", "grip"])
            e = e[[(t, bp) in sub for t, bp in zip(e.task, e.base)]]
            if not len(e):
                continue
            e["lg"] = 8.123 + 0.4445 * e.z.astype(float) + 11.3193 * (-e.grip.astype(float))
            d = [lg - base_lg[(t, bp)] for t, bp, lg in zip(e.task, e.base, e.lg)
                 if (t, bp) in base_lg]
            if d:
                pts.append((it, sum(d) / len(d)))
        out[name] = pts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args()
    run = ROOT / "results/rules_runs" / args.run_id

    series = {}   # pass -> metric -> [(iter, delta)]
    for pdir in sorted(run.glob("pass_*")):
        name = pdir.name.split("_", 1)[1]
        s = {m: [] for m, _ in PANELS if m != "val8r"}
        for f in sorted(pdir.glob("iter_*/scores.json")):
            it = int(f.parent.name.split("_")[1])
            d = json.load(open(f))["delta"]
            for m, _ in PANELS:
                if m != "val8r" and d.get(m) is not None:
                    s[m].append((it, d[m]))
        # partial iterations: train-only points
        for f in sorted(pdir.glob("iter_*/eval_train.json")):
            it = int(f.parent.name.split("_")[1])
            if not (f.parent / "scores.json").exists():
                s["train"].append((it, json.load(open(f))["delta"]))
        series[name] = s
    v8r = val8_recomposed_series(run)
    for name in series:
        series[name]["val8r"] = v8r.get(name, [])

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), sharex=True)
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
