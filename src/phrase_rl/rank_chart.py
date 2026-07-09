"""Ranked rephrase chart: base-Qwen vs tuned (phase2 best_val) vs original, by CRN flow reward.

Data: results/overnight/raw/scores_rankchart.parquet (rankchart pipeline, pod 1).
Per task, every phrase's mean flow loss (over Bridge-matched contexts x 16 CRN draws),
ranked best->worst. Color = source arm.

  .venv/bin/python -m phrase_rl.rank_chart
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"base": "#378ADD", "tuned": "#D85A30", "original": "#222222"}


def main():
    sc = pd.read_parquet("results/overnight/raw/scores_rankchart.parquet")
    tasks = sorted(sc.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 6.2), sharex=False)

    for ax, task in zip(axes, tasks):
        g = (sc[sc.task == task].groupby(["phrase", "arm"])["loss"].mean()
             .reset_index().sort_values("loss", ascending=True).reset_index(drop=True))
        for i, r in g.iterrows():
            ax.scatter(r.loss, i, s=42 if r.arm != "original" else 130,
                       marker="o" if r.arm != "original" else "*",
                       color=COLORS[r.arm], zorder=3 if r.arm != "original" else 4)
        n = len(g)
        med = {arm: g.index[g.arm == arm].to_series().median() for arm in ("base", "tuned")}
        ax.set_title(f"{task.replace('widowx_', '')}\nmedian rank: tuned {med['tuned']:.0f} vs base {med['base']:.0f} (of {n})",
                     fontsize=10)
        ax.set_ylim(n, -1)  # best (rank 0) on top
        ax.set_ylabel("rank by flow reward (best on top)" if task == tasks[0] else "")
        ax.set_xlabel("mean flow loss (lower = better reward)")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    for arm, c in COLORS.items():
        axes[0].scatter([], [], color=c, marker="*" if arm == "original" else "o",
                        s=130 if arm == "original" else 42, label=arm)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Rephrases ranked by frozen-pi0 flow reward — base Qwen vs tuned (flow arm, best_val@200), trace-conditioned sampling",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("results/charts/rephrase_rank_base_vs_tuned.png", dpi=130, bbox_inches="tight")
    print("chart -> results/charts/rephrase_rank_base_vs_tuned.png")

    # side table for the chat/writeup: top-5 per task with arm tags
    for task in tasks:
        g = (sc[sc.task == task].groupby(["phrase", "arm"])["loss"].mean()
             .reset_index().sort_values("loss").head(5))
        print(f"\n== {task} top-5 by reward ==")
        for r in g.itertuples():
            print(f"  [{r.arm:8s}] {r.loss:.4f}  {r.phrase[:75]!r}")


if __name__ == "__main__":
    main()
