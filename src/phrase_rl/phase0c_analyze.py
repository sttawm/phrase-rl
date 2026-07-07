"""Phase 0c money plot: does lower flow loss predict higher rollout success?

Joins per-(task, phrase) flow loss (averaged over matched contexts + draws) with
per-(task, phrase) rollout success (averaged over shared episode_ids). Reports,
per task and pooled: Spearman(reward = -loss, success). Positive = the reward
ranks phrases the way rollout success does.

  python -m phrase_rl.phase0c_analyze \
    --scores data/scores_0c.parquet --rollouts data/rollouts_0c.parquet --outdir results/phase0c
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--outdir", default="results/phase0c")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("results/charts", exist_ok=True)

    sc = pd.read_parquet(args.scores)
    ro = pd.read_parquet(args.rollouts)

    # per (task, phrase): mean flow loss (over contexts+draws) and mean success (over episode_ids)
    loss = sc.groupby(["task", "phrase"])["loss"].mean().rename("flow_loss")
    succ = ro.groupby(["task", "phrase"]).agg(
        success=("success", "mean"), grasped=("grasped", "mean"), n_ep=("success", "size")
    )
    df = pd.concat([loss, succ], axis=1).dropna(subset=["flow_loss", "success"]).reset_index()
    df["reward"] = -df["flow_loss"]

    summary = {"per_task": {}, "n_tasks_with_flow_loss": int(df["task"].nunique())}
    tasks = sorted(df["task"].unique())
    fig, axes = plt.subplots(1, len(tasks) + 1, figsize=(5 * (len(tasks) + 1), 4.2))
    if len(tasks) == 0:
        axes = [axes]

    pooled_rank_reward, pooled_rank_succ = [], []
    for i, task in enumerate(tasks):
        g = df[df.task == task]
        # correlation on success; also on a graded score (success + 0.3*grasp) for resolution at n=10
        graded = g["success"] + 0.3 * g["grasped"]
        rho_s = stats.spearmanr(g["reward"], g["success"]).statistic
        rho_g = stats.spearmanr(g["reward"], graded).statistic
        summary["per_task"][task] = {
            "n_phrases": int(len(g)),
            "spearman_reward_success": round(float(rho_s), 3),
            "spearman_reward_graded": round(float(rho_g), 3),
            "success_range": [round(float(g.success.min()), 2), round(float(g.success.max()), 2)],
            "flow_loss_range": [round(float(g.flow_loss.min()), 4), round(float(g.flow_loss.max()), 4)],
        }
        pooled_rank_reward.extend(stats.rankdata(g["reward"]))
        pooled_rank_succ.extend(stats.rankdata(graded))
        axes[i].scatter(g["reward"], graded, s=18, alpha=0.6)
        axes[i].set(title=f"{task.replace('widowx_', '')}\nρ(succ)={rho_s:+.2f} ρ(graded)={rho_g:+.2f}",
                    xlabel="reward (−flow loss)", ylabel="success + 0.3·grasp")

    if pooled_rank_reward:
        rho_pool = stats.spearmanr(pooled_rank_reward, pooled_rank_succ).statistic
        summary["pooled_spearman_withintask_ranked"] = round(float(rho_pool), 3)
        axes[-1].scatter(pooled_rank_reward, pooled_rank_succ, s=14, alpha=0.5)
        axes[-1].set(title=f"pooled (within-task ranks)\nρ={rho_pool:+.2f}",
                     xlabel="reward rank", ylabel="graded-success rank")
    fig.suptitle("Phase 0c — does flow-loss reward predict SIMPLER rollout success?", y=1.02)
    fig.tight_layout()
    chart = "results/charts/phase0c_loss_vs_success.png"
    fig.savefig(chart, dpi=130, bbox_inches="tight")
    summary["chart"] = chart

    # overall rollout sensitivity (all tasks incl. eggplant, which lacks flow loss)
    rollout_by_task = {}
    for t, g in ro.groupby("task"):
        per_phrase = g.groupby("phrase")["success"].mean()
        rollout_by_task[t] = {
            "mean_success": round(float(g["success"].mean()), 3),
            "phrase_success_std": round(float(per_phrase.std()), 3),
            "n_phrases": int(per_phrase.size),
        }
    summary["rollout_success_by_task"] = rollout_by_task
    df.to_parquet(os.path.join(args.outdir, "per_phrase.parquet"), index=False)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
