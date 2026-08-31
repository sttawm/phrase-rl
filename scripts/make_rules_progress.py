#!/usr/bin/env python3
"""results/charts/rules_progress_<run>.png -- live rules-loop progress.

Top: pipeline grid -- pre-loop stages, then iterations x 6 phases (distill,
apply, score, judge, validate, probe), colored by state, derived purely from
run-dir artifacts. Bottom: grouped bars of the eval metric per iteration
(train / val_held / val8) plus deltas vs the unrephrased baseline.

  .venv/bin/python scripts/make_rules_progress.py r1 [--pass gemini] [--demo]
"""
import argparse
import glob
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DONE, ACTIVE, PEND = "#0f766e", "#f59e0b", "#e5e7eb"
PHASES = ["distill", "apply", "score", "judge", "validate", "probe"]


def phase_states(run_dir, pdir):
    """One row per iteration; each cell done/active/pending from artifacts."""
    rows = []
    idirs = sorted(pdir.glob("iter_*"))
    for it, idir in enumerate(idirs):
        st = {}
        st["distill"] = (idir / "rules.md").exists()
        ev = list(idir.glob("eval_train_*.parquet"))
        st["score"] = bool(ev)
        # applies precede scoring; if scored, applied. If not scored but the
        # iteration dir exists and rules do, apply/score are the active front.
        st["apply"] = st["score"] or st["distill"]
        st["judge"] = (idir / "judge.md").exists()
        st["validate"] = (idir / "eval_val8.json").exists() and \
                         (idir / "eval_val_held.json").exists()
        st["probe"] = (idir / "scores.json").exists()  # probes follow scoring bookkeeping
        rows.append(st)
    grid = []
    for i, st in enumerate(rows):
        row = []
        seen_active = False
        for ph in PHASES:
            if st[ph]:
                row.append(DONE)
            elif not seen_active:
                row.append(ACTIVE)
                seen_active = True
            else:
                row.append(PEND)
        grid.append(row)
    return grid


def preloop_states(run_dir):
    baseline = bool(glob.glob(str(run_dir / "baseline_*_logit.json")))
    return [
        ("score bank", (run_dir / "bank.parquet").exists()),
        ("corpus stats", (run_dir / "corpus_stats.md").exists()),
        ("splits + base pool", (run_dir / "splits.json").exists()
         and (run_dir / "base_pool.parquet").exists()),
        ("baselines", baseline),
    ]


def demo_data():
    rng = np.random.default_rng(3)
    scores = []
    base = {"train": 0.55, "val_held": 0.40, "val8": 0.62}
    for it in range(5):
        gain = 0.0 if it == 0 else 0.12 * it - 0.02 * it * it
        scores.append({k: v + gain + rng.normal(0, 0.04) for k, v in base.items()})
    grid = [[DONE] * 6 for _ in range(4)] + [[DONE, DONE, ACTIVE, PEND, PEND, PEND]]
    pre = [(n, True) for n, _ in
           [("score bank", 1), ("corpus stats", 1), ("splits + base pool", 1), ("baselines", 1)]]
    deltas = [{k: s[k] - base[k] for k in s} for s in scores]
    return pre, grid, scores, deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--pass", dest="pass_", default="gemini")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        pre, grid, scores, deltas = demo_data()
        tag = f"{args.run_id}-DEMO"
    else:
        run_dir = ROOT / "results/rules_runs" / args.run_id
        pdir = run_dir / f"pass_{args.pass_}"
        pre = preloop_states(run_dir)
        grid = phase_states(run_dir, pdir)
        scores, deltas = [], []
        for f in sorted(pdir.glob("iter_*/scores.json")):
            d = json.load(open(f))
            scores.append({k: d[k] for k in ("train", "val_held", "val8")})
            deltas.append(d.get("delta") or {})
        tag = f"{args.run_id}/{args.pass_}"

    n_it = max(len(grid), 1)
    fig = plt.figure(figsize=(11.5, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.0, 1.0],
                          hspace=0.42, wspace=0.25)

    # --- pipeline grid ---------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    for j, (name, ok) in enumerate(pre):
        ax.add_patch(plt.Rectangle((j, n_it + 0.6), 0.92, 0.8,
                                   color=DONE if ok else ACTIVE))
        ax.text(j + 0.46, n_it + 1.0, name, ha="center", va="center", fontsize=7.5,
                color="white" if ok else "#78350f")
    for i, row in enumerate(grid):
        y = n_it - 1 - i
        for j, color in enumerate(row):
            ax.add_patch(plt.Rectangle((j, y), 0.92, 0.8, color=color))
        ax.text(-0.15, y + 0.4, f"iter {i}", ha="right", va="center", fontsize=8)
    for j, ph in enumerate(PHASES):
        ax.text(j + 0.46, -0.35, ph, ha="center", va="top", fontsize=8.5)
    ax.text(-0.15, n_it + 1.0, "pre-loop", ha="right", va="center", fontsize=8,
            fontstyle="italic")
    ax.set_xlim(-1.4, max(len(PHASES), len(pre)) + 0.2)
    ax.set_ylim(-1.0, n_it + 1.7)
    ax.axis("off")
    ax.set_title(f"rules loop {tag} -- pipeline", fontsize=11)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (DONE, ACTIVE, PEND)],
              labels=["done", "in progress", "pending"], fontsize=7.5, ncol=3,
              loc="upper right", frameon=False)

    # --- score bars -------------------------------------------------------
    axb = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])
    COLS = {"train": "#94a3b8", "val_held": "#2563eb", "val8": "#7c3aed"}
    if scores:
        x = np.arange(len(scores))
        for k, off in (("train", -0.27), ("val_held", 0.0), ("val8", 0.27)):
            axb.bar(x + off, [s[k] for s in scores], 0.25, color=COLS[k], label=k)
            if deltas and all(d.get(k) is not None for d in deltas):
                axd.bar(x + off, [d[k] for d in deltas], 0.25, color=COLS[k], label=k)
        axd.axhline(0, color="#6b7280", lw=1)
        for a2, ttl, yl in ((axb, "eval metric per iteration", "mean proxy logit"),
                            (axd, "delta vs unrephrased bases", "logit gain")):
            a2.set_xticks(x)
            a2.set_xticklabels([f"it{j}" for j in x], fontsize=8)
            a2.set_title(ttl, fontsize=10)
            a2.set_ylabel(yl, fontsize=9)
            a2.grid(alpha=0.25, axis="y")
        axb.legend(fontsize=7.5, frameon=False)
    else:
        for a2 in (axb, axd):
            a2.text(0.5, 0.5, "no scored iterations yet", ha="center", fontsize=10,
                    color="#6b7280", transform=a2.transAxes)
            a2.axis("off")

    out = ROOT / "results/charts" / f"rules_progress_{tag.replace('/', '_')}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.25)
    print(f"chart -> {out}")


if __name__ == "__main__":
    main()
