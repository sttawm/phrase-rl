"""Reward bake-off chart: LOTO Spearman per arm (per-task dots + mean bar).

  python -m phrase_rl.bakeoff_chart
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def build_matrix():
    sc = pd.read_parquet("data/scores_0c_redo.parquet")
    dec = pd.read_parquet("data/decoded_0c_redo.parquet")
    ro = pd.concat([pd.read_parquet("data/rollouts_0c_redo.parquet"),
                    pd.read_parquet("data/rollouts_deepseed.parquet")])
    fa = pd.read_parquet("results/phase0c/faithfulness_redo.parquet")
    gate_ok = set(map(tuple, fa[fa.faithful][["task", "phrase"]].itertuples(index=False)))
    lab = ro.groupby(["task", "phrase"]).agg(success=("success", "mean"), grasped=("grasped", "mean"), n=("success", "size"))
    lab["graded"] = lab["success"] + 0.3 * lab["grasped"]
    rows = []
    for (task, ep, t), g in sc.groupby(["task", "episode_index", "t"]):
        piv = g.pivot_table(index="phrase", columns="k", values="loss")
        z = (piv - piv.mean(axis=0)) / (piv.std(axis=0) + 1e-9)
        z["task"] = task
        rows.append(z.reset_index())
    tau = pd.concat(rows).groupby(["task", "phrase"])[list(range(16))].mean()
    raw = sc.groupby(["task", "phrase"])["loss"].mean().rename("flow_raw")
    dl2 = dec.groupby(["task", "phrase"])[["norm_l2", "grip_err"]].mean()
    df = tau.join(raw).join(dl2).join(lab).dropna(subset=["success", "flow_raw"])
    df = df[[i in gate_ok for i in df.index]]
    return df


def main():
    df = build_matrix()
    tasks = df.index.get_level_values(0).values
    y = df["graded"].values
    X16 = df[list(range(16))].values
    bands = np.stack([X16[:, b:b + 4].mean(1) for b in range(0, 16, 4)], 1)
    arms = {
        "raw flow loss": lambda tr, te: -df["flow_raw"].values[te],
        "flow (z-scored)": lambda tr, te: -X16[te].mean(1),
        "best τ-band": None,  # set below
        "decoded-L2": lambda tr, te: -df["norm_l2"].values[te],
    }

    def best_band(tr, te):
        best, bi = -2, 0
        for i in range(4):
            r = stats.spearmanr(-bands[tr, i], y[tr]).statistic
            if r > best: best, bi = r, i
        return -bands[te, bi]
    arms["best τ-band"] = best_band

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    names = list(arms)
    task_names = sorted(np.unique(tasks))
    markers = {t: m for t, m in zip(task_names, ["o", "s", "^"])}
    for xi, name in enumerate(names):
        rs = []
        for held in task_names:
            tr, te = tasks != held, tasks == held
            r = stats.spearmanr(arms[name](tr, te), y[te]).statistic
            rs.append(r)
            ax.scatter([xi], [r], marker=markers[held], s=48, color="#378ADD", alpha=0.75, zorder=3)
        ax.hlines(np.mean(rs), xi - 0.28, xi + 0.28, color="#D85A30", lw=3, zorder=4)
        ax.text(xi, np.mean(rs) + 0.03, f"{np.mean(rs):+.2f}", ha="center", fontsize=9, color="#993C1D")
    ax.axhline(0, color="#B4B2A9", lw=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Spearman ρ vs rollout success\n(held-out task)")
    ax.set_title("Reward bake-off — leave-one-task-out validity, gated CoVer-template phrases (n=94, 3 tasks)", fontsize=11)
    for t in task_names:
        ax.scatter([], [], marker=markers[t], color="#378ADD", label=t.replace("widowx_", ""))
    ax.scatter([], [], marker="_", color="#D85A30", label="LOTO mean")
    ax.legend(fontsize=8, loc="lower right")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/charts/bakeoff.png", dpi=130, bbox_inches="tight")
    print("chart -> results/charts/bakeoff.png")


if __name__ == "__main__":
    main()
