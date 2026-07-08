"""Phase 0b analysis: is the CRN flow-loss reward reliable at training-budget K?

Inputs: long-format score parquets from phase0b_score (one per checkpoint).
Per (checkpoint, arm, context):
  - split-half reliability: even-k vs odd-k draw halves (tau-stratified draws
    interleave, so halves have matched tau coverage) -> Spearman across phrases
  - raw vs per-draw z-scored loss variants
  - variance decomposition: noise var from half differences, signal share
  - Spearman-Brown projection from K/2-half reliability to full-K reliability
Plus: phrase-length/loss correlation (hackable axis), discriminability vs tau,
reliability vs episode position, arm-level loss stats, original-vs-rephrase.

Usage:
  python -m phrase_rl.phase0b_analyze \
    --scores rephrase=data/scores_0b_rephrase.parquet plain=data/scores_0b_plain.parquet \
    --outdir results/phase0b
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ARMS_ANALYZED = None  # None = every arm in the data except "original"


def context_matrices(df):
    """Yield (episode_index, t, arm, loss_matrix P x K, phrases) per context+arm."""
    arms = ARMS_ANALYZED or [a for a in df["arm"].unique() if a != "original"]
    for (ep, t), g in df.groupby(["episode_index", "t"]):
        for arm in arms:
            ga = g[g["arm"] == arm]
            if ga["phrase"].nunique() < 8:
                continue
            mat = ga.pivot_table(index="phrase", columns="k", values="loss")
            yield ep, t, arm, mat.to_numpy(), list(mat.index)


def split_half_stats(mat, zscore=False):
    """mat: (P, K). Returns spearman(even, odd), signal_share, spearman_brown_fullK."""
    m = mat.copy()
    if zscore:
        mu, sd = m.mean(axis=0, keepdims=True), m.std(axis=0, keepdims=True) + 1e-9
        m = (m - mu) / sd
    a, b = m[:, 0::2].mean(axis=1), m[:, 1::2].mean(axis=1)
    rho = stats.spearmanr(a, b).statistic
    noise_var = np.var(a - b, ddof=1) / 2.0
    total_var = np.var((a + b) / 2.0, ddof=1)
    signal_share = max(0.0, 1.0 - (noise_var / 2.0) / max(total_var, 1e-12))
    sb_full = 2 * rho / (1 + rho) if rho > -1 else np.nan
    return rho, signal_share, sb_full


def analyze_file(name, path, outdir):
    df = pd.read_parquet(path)
    rows, tau_disc, len_rows = [], [], []
    for ep, t, arm, mat, phrases in context_matrices(df):
        rho_raw, sig_raw, sb_raw = split_half_stats(mat, zscore=False)
        rho_z, sig_z, sb_z = split_half_stats(mat, zscore=True)
        phrase_means = mat.mean(axis=1)
        wc = np.array([len(p.split()) for p in phrases])
        len_r = stats.spearmanr(wc, phrase_means).statistic if len(set(wc)) > 1 else np.nan
        rows.append(
            dict(episode_index=ep, t=t, arm=arm, n_phrases=len(phrases),
                 rho_raw=rho_raw, rho_z=rho_z, sig_raw=sig_raw, sig_z=sig_z,
                 sb_raw=sb_raw, sb_z=sb_z, len_corr=len_r,
                 spread=float(np.std(phrase_means)), mean_loss=float(np.mean(phrase_means)))
        )
        # per-draw discriminability: corr of draw-k phrase losses w/ leave-one-out mean
        K = mat.shape[1]
        for k in range(K):
            rest = mat[:, [j for j in range(K) if j != k]].mean(axis=1)
            r = stats.spearmanr(mat[:, k], rest).statistic
            tau_disc.append(dict(arm=arm, k=k, rho=r))
        len_rows.append(len_r)

    per_ctx = pd.DataFrame(rows)
    tau_df = pd.DataFrame(tau_disc)
    tau_map = df.groupby("k")["tau"].first()

    summary = {}
    for arm, g in per_ctx.groupby("arm"):
        summary[arm] = {
            "n_contexts": int(len(g)),
            "split_half_spearman_raw": {"median": float(g.rho_raw.median()), "q25": float(g.rho_raw.quantile(.25)), "q75": float(g.rho_raw.quantile(.75))},
            "split_half_spearman_z": {"median": float(g.rho_z.median()), "q25": float(g.rho_z.quantile(.25)), "q75": float(g.rho_z.quantile(.75))},
            "spearman_brown_fullK_median": float(g.sb_z.median()),
            "signal_share_z_median": float(g.sig_z.median()),
            "length_loss_corr_median": float(np.nanmedian(g.len_corr)),
            "phrase_spread_median": float(g.spread.median()),
            "mean_loss_median": float(g.mean_loss.median()),
        }

    # original vs rephrase oracle gain
    orig = df[df.arm == "original"].groupby(["episode_index", "t"])["loss"].mean()
    reph = df[df.arm != "original"].groupby(["episode_index", "t", "phrase"])["loss"].mean()
    best = reph.groupby(["episode_index", "t"]).min()
    joined = pd.concat([orig.rename("orig"), best.rename("best")], axis=1).dropna()
    summary["oracle"] = {
        "frac_contexts_best_rephrase_beats_original": float((joined.best < joined.orig).mean()),
        "median_relative_gain": float(((joined.orig - joined.best) / joined.orig).median()),
    }

    # charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for arm, g in per_ctx.groupby("arm"):
        axes[0].hist(g.rho_z.dropna(), bins=30, alpha=0.5, label=arm)
    axes[0].set(title=f"{name}: split-half Spearman (z-scored)", xlabel="rho per context"); axes[0].legend()
    td = tau_df.merge(tau_map.rename("tau"), left_on="k", right_index=True)
    for arm, g in td.groupby("arm"):
        m = g.groupby("tau")["rho"].median().sort_index()
        axes[1].plot(m.index, m.values, marker="o", label=arm)
    axes[1].set(title="discriminability vs tau (1=noise end)", xlabel="tau", ylabel="draw-vs-rest rho"); axes[1].legend()
    for arm, g in per_ctx.groupby("arm"):
        axes[2].scatter(g.len_corr, g.rho_z, s=8, alpha=0.5, label=arm)
    axes[2].set(title="length-loss corr vs reliability", xlabel="length corr", ylabel="rho_z"); axes[2].legend()
    fig.tight_layout()
    chart = os.path.join("results/charts", f"phase0b_{name}.png")
    fig.savefig(chart, dpi=120)
    summary["chart"] = chart

    per_ctx.to_parquet(os.path.join(outdir, f"per_context_{name}.parquet"), index=False)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--outdir", default="results/phase0b")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("results/charts", exist_ok=True)

    out = {}
    for spec in args.scores:
        name, path = spec.split("=", 1)
        out[name] = analyze_file(name, path, args.outdir)
        print(f"\n=== {name} ===")
        print(json.dumps(out[name], indent=2))

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.outdir}/metrics.json")


if __name__ == "__main__":
    main()
