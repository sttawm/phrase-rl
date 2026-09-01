#!/usr/bin/env python3
"""results/charts/rules_by_kind_<run>.png -- per-regime learning curves.

Grid: rows = base kind (natural / adversarial / original), cols = eval set
(train / val_held / val8). Each cell: delta vs the unrephrased base per
iteration, one line per rephraser. Iterations whose eval parquet was lost are
rebuilt from the apply+score job results.

  .venv/bin/python scripts/make_rules_by_kind.py r1
"""
import argparse
import glob
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
COLORS = {"gemini": "#2563eb", "claude": "#d97706", "qwen": "#059669"}
KINDS = ["natural", "adversarial", "original"]
TAGS = ["train", "val_held", "val8"]


def logit(df):
    return 8.123 + 0.4445 * df.z.astype(float) + 11.3193 * (-df.grip.astype(float))


def pairs_from_jobs(run, tag, sha):
    for spec in glob.glob(str(run / "jobs" / f"{tag}_*.spec.json")):
        sp = json.load(open(spec))
        if sp.get("kind") != "apply" or not str(sp.get("rules_sha", "")).startswith(sha):
            continue
        ap = pathlib.Path(spec.replace(".spec.json", ".result.parquet"))
        if not ap.exists():
            continue
        a = pd.read_parquet(ap)
        for sspec in glob.glob(str(run / "jobs" / f"{tag}_sc_*.spec.json")):
            rp = pathlib.Path(sspec.replace(".spec.json", ".result.parquet"))
            if not rp.exists():
                continue
            r = pd.read_parquet(rp)
            if len(set(r.phrase) & set(a.rewrite)) / max(len(r), 1) > 0.9:
                m = r.merge(a.rename(columns={"phrase": "base"}),
                            left_on=["task", "phrase"], right_on=["task", "rewrite"])
                return m[["task", "base", "z", "grip"]]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args()
    run = ROOT / "results/rules_runs" / args.run_id

    bank = pd.read_parquet(run / "bank.parquet")
    kinds = {(t, p): k for t, p, k in
             zip(bank.task, bank.phrase, bank.kind.fillna("unknown"))}

    base_lg = {}
    for tag in TAGS:
        rows = list(run.glob(f"baseline_{tag}_*_rows.parquet"))
        if rows:
            b = pd.read_parquet(rows[0]).dropna(subset=["z", "grip"])
            base_lg[tag] = {(t, p): v for t, p, v in zip(b.task, b.phrase, logit(b))}

    # series[tag][kind][pass] = [(iter, mean delta)]
    series = {t: {k: {} for k in KINDS} for t in TAGS}
    for pdir in sorted(run.glob("pass_*")):
        name = pdir.name.split("_", 1)[1]
        for tag in TAGS:
            per_iter = {}
            for f in sorted(pdir.glob(f"iter_*/eval_{tag}_*.parquet")):
                it = int(f.parent.name.split("_")[1])
                per_iter[it] = pd.read_parquet(f)
            for f in sorted(pdir.glob(f"iter_*/eval_{tag}_*.json")):
                if f.name == f"eval_{tag}.json":
                    continue
                it = int(f.parent.name.split("_")[1])
                if it not in per_iter:
                    e = pairs_from_jobs(run, tag, f.stem.split(f"eval_{tag}_")[1])
                    if e is not None:
                        per_iter[it] = e
            for it, e in per_iter.items():
                e = e.dropna(subset=["z", "grip"]).copy()
                if "base" not in e or tag not in base_lg:
                    continue
                e["lg"] = logit(e)
                e["kind"] = [kinds.get((t, b), "unknown") for t, b in zip(e.task, e.base)]
                e["delta"] = [v - base_lg[tag].get((t, b), float("nan"))
                              for t, b, v in zip(e.task, e.base, e.lg)]
                e = e.dropna(subset=["delta"])
                for k in KINDS:
                    g = e[e.kind == k]
                    if len(g):
                        series[tag][k].setdefault(name, []).append(
                            (it, float(g.delta.mean()), len(g)))

    fig, axes = plt.subplots(len(KINDS), len(TAGS), figsize=(13.5, 10.5),
                             sharex=True)
    for i, k in enumerate(KINDS):
        for j, tag in enumerate(TAGS):
            ax = axes[i][j]
            cell = series[tag][k]
            n_note = ""
            for name, pts in sorted(cell.items()):
                pts = sorted(pts)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                        color=COLORS.get(name, "#6b7280"), label=name, lw=1.8, ms=4)
                n_note = f"n≈{pts[-1][2]}"
            ax.axhline(0, color="#6b7280", lw=0.8, ls="--")
            ax.grid(alpha=0.25)
            if not cell:
                ax.text(0.5, 0.5, "no phrases of this kind", ha="center",
                        va="center", transform=ax.transAxes, color="#9ca3af")
            else:
                ax.text(0.02, 0.95, n_note, transform=ax.transAxes, fontsize=7.5,
                        va="top", color="#6b7280")
            if i == 0:
                ax.set_title(tag, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{k}\ndelta (mean logit)", fontsize=10)
            if i == len(KINDS) - 1:
                ax.set_xlabel("iteration")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0][0].legend(fontsize=9, frameon=False)
    fig.suptitle(f"rules loop {args.run_id} -- per-regime deltas "
                 f"(rows: base kind, cols: eval set)", fontsize=12)
    fig.tight_layout()
    out = ROOT / "results/charts" / f"rules_by_kind_{args.run_id}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"chart -> {out}")


if __name__ == "__main__":
    main()
