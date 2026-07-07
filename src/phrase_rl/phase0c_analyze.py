"""Phase 0c analysis. Two separable questions, only one crosses the real→sim gap:

(A) CLEAN, all-in-sim: does phrasing move sim success? Do rephrasings beat the
    original instruction in sim? No real/sim confound — phrases and success are
    both in sim. This is the headline.
(B) CAVEATED bonus: does the real-frame flow-loss reward rank phrases the way sim
    success does? Crosses real→sim, so a positive is informative but a null is
    uninterpretable (broken reward vs. domain-specific phrasing).

(A) uses all 4 tasks (rollouts only). (B) uses the 3 tasks with matched real
Bridge flow-loss (eggplant excluded — no Bridge counterpart).

  python -m phrase_rl.phase0c_analyze \
    --rollouts data/rollouts_0c.parquet --scores data/scores_0c.parquet --outdir results/phase0c
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


def split_half_success_rho(g):
    """Rank phrases by success on even vs odd episode_ids; Spearman. Noisy at n≈5/half."""
    ev = g[g.episode_id % 2 == 0].groupby("phrase")["success"].mean()
    od = g[g.episode_id % 2 == 1].groupby("phrase")["success"].mean()
    j = pd.concat([ev.rename("e"), od.rename("o")], axis=1).dropna()
    if len(j) < 5 or j.e.std() == 0 or j.o.std() == 0:
        return None
    return float(stats.spearmanr(j.e, j.o).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--scores", default=None)
    ap.add_argument("--outdir", default="results/phase0c")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("results/charts", exist_ok=True)

    ro = pd.read_parquet(args.rollouts)
    summary = {"A_in_sim": {}, "B_reward_vs_success": {}}

    # ---- (A) in-sim phrasing sensitivity — the clean headline ----
    tasks = sorted(ro["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4.6), squeeze=False)
    axes = axes[0]
    for i, task in enumerate(tasks):
        g = ro[ro.task == task]
        per = g.groupby(["arm", "phrase"])["success"].mean()
        orig = float(per.xs("original", level="arm").iloc[0]) if "original" in per.index.get_level_values("arm") else np.nan
        reph = per.drop("original", level="arm") if "original" in per.index.get_level_values("arm") else per
        summary["A_in_sim"][task] = {
            "n_phrases": int(per.size),
            "n_episodes": int(len(g)),
            "original_success": round(orig, 3),
            "rephrase_success_mean": round(float(reph.mean()), 3),
            "rephrase_success_best": round(float(reph.max()), 3),
            "frac_rephrasings_beat_original": round(float((reph > orig).mean()), 3),
            "success_std_across_phrases": round(float(per.std()), 3),
            "splithalf_rho_success": split_half_success_rho(g),
        }
        # money plot: per-phrase success distribution, original marked
        vals = np.sort(reph.values)
        axes[i].scatter(range(len(vals)), vals, s=16, color="#85B7EB", label="rephrasings")
        axes[i].axhline(orig, color="#D85A30", lw=2, label=f"original ({orig:.2f})")
        axes[i].axhline(reph.mean(), color="#5F5E5A", ls="--", lw=1, label=f"rephrase mean ({reph.mean():.2f})")
        axes[i].set(title=task.replace("widowx_", ""), xlabel="phrase (sorted)", ylabel="sim success")
        axes[i].legend(fontsize=7, loc="upper left")
    fig.suptitle("Phase 0c (A) — in-sim: do rephrasings beat the original instruction? [no real/sim confound]", y=1.02)
    fig.tight_layout()
    chartA = "results/charts/phase0c_insim_phrasing.png"
    fig.savefig(chartA, dpi=130, bbox_inches="tight")
    summary["A_in_sim_chart"] = chartA

    # ---- (B) reward (real flow loss) vs sim success — caveated ----
    if args.scores and os.path.exists(args.scores):
        sc = pd.read_parquet(args.scores)
        loss = sc.groupby(["task", "phrase"])["loss"].mean().rename("flow_loss")
        succ = ro.groupby(["task", "phrase"]).agg(success=("success", "mean"), grasped=("grasped", "mean"))
        df = pd.concat([loss, succ], axis=1).dropna(subset=["flow_loss", "success"]).reset_index()
        df["reward"] = -df["flow_loss"]
        btasks = sorted(df["task"].unique())
        figB, axB = plt.subplots(1, len(btasks) + 1, figsize=(4.5 * (len(btasks) + 1), 4), squeeze=False)
        axB = axB[0]
        pr, ps = [], []
        for i, task in enumerate(btasks):
            gg = df[df.task == task]
            graded = gg["success"] + 0.3 * gg["grasped"]
            rho_s = stats.spearmanr(gg["reward"], gg["success"]).statistic
            rho_g = stats.spearmanr(gg["reward"], graded).statistic
            summary["B_reward_vs_success"][task] = {
                "n_phrases": int(len(gg)),
                "spearman_reward_success": round(float(rho_s), 3),
                "spearman_reward_graded": round(float(rho_g), 3),
            }
            pr.extend(stats.rankdata(gg["reward"])); ps.extend(stats.rankdata(graded))
            axB[i].scatter(gg["reward"], graded, s=18, alpha=0.6)
            axB[i].set(title=f"{task.replace('widowx_','')}\nρ={rho_s:+.2f} (graded {rho_g:+.2f})",
                       xlabel="reward (−flow loss)", ylabel="success+0.3·grasp")
        if pr:
            rp = stats.spearmanr(pr, ps).statistic
            summary["B_reward_vs_success"]["pooled_withintask_ranked"] = round(float(rp), 3)
            axB[-1].scatter(pr, ps, s=14, alpha=0.5)
            axB[-1].set(title=f"pooled within-task ranks\nρ={rp:+.2f}", xlabel="reward rank", ylabel="success rank")
        figB.suptitle("Phase 0c (B) — real-frame reward vs sim success [CAVEATED: crosses real→sim; +ve informative, null not]", y=1.02)
        figB.tight_layout()
        chartB = "results/charts/phase0c_reward_vs_success.png"
        figB.savefig(chartB, dpi=130, bbox_inches="tight")
        summary["B_chart"] = chartB
        df.to_parquet(os.path.join(args.outdir, "per_phrase.parquet"), index=False)

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
