"""Render Phase 2 training curves from train_log.jsonl -> results/charts/phase2_training.png.

  python -m phrase_rl.phase2_curves --log data/phase2_train_log.jsonl
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="results/charts/phase2_training.png")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.log) if l.strip()]
    steps = pd.DataFrame([r for r in rows if "sec" in r and r.get("n_ok")])
    vals = pd.DataFrame([r for r in rows if r.get("type") == "val"])

    def smooth(x, w=15):
        return pd.Series(x).rolling(w, min_periods=1).mean()

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))

    ax[0].plot(steps["step"], smooth(-steps["cand_loss_mean"]), color="#378ADD", label="candidates (mean)")
    ax[0].plot(steps["step"], smooth(-steps["orig_loss_mean"]), color="#5F5E5A", ls="--", label="original instruction")
    ax[0].set(title="reward (−flow loss), train contexts", xlabel="step"); ax[0].legend(fontsize=8)

    if len(vals):
        ax[1].plot(vals["step"], vals["mean_reward"], marker="o", color="#378ADD", label="val mean")
        ax[1].plot(vals["step"], vals["mean_best_reward"], marker="s", color="#1D9E75", label="val best-of-16")
        ax[1].axhline(vals["mean_orig_reward"].iloc[-1], color="#5F5E5A", ls="--", label="val original")
        ax[1].legend(fontsize=8)
    else:
        ax[1].text(0.5, 0.5, "val: first eval at step 100", ha="center", va="center", transform=ax[1].transAxes)
    ax[1].set(title="validation reward", xlabel="step")

    ax[2].plot(steps["step"], smooth(steps["gate_pass_rate"]), color="#1D9E75", label="gate pass")
    ax[2].plot(steps["step"], smooth(steps["rename_rate"]), color="#BA7517", label="rename rate")
    ax[2].plot(steps["step"], smooth(steps["parse_fail_rate"]), color="#A32D2D", label="parse fail")
    ax[2].set(title="generation health", xlabel="step", ylim=(0, 1)); ax[2].legend(fontsize=8)

    ax[3].plot(steps["step"], smooth(steps["kl"].fillna(0)), color="#7F77DD", label="KL to base")
    ax3b = ax[3].twinx()
    ax3b.plot(steps["step"], smooth(steps["n_pos"]), color="#D85A30", alpha=0.6, label="n positive")
    ax[3].set(title="KL (left) / positive candidates (right)", xlabel="step")
    ax[3].legend(loc="upper left", fontsize=8); ax3b.legend(loc="upper right", fontsize=8)

    for a in ax:
        for sp in ["top", "right"]: a.spines[sp].set_visible(False)
    fig.suptitle(f"Phase 2 RL — trace-conditioned, Qwen judge, source-aug (through step {int(steps['step'].max())})", y=1.03)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"chart -> {args.out} ({len(steps)} steps, {len(vals)} vals)")


if __name__ == "__main__":
    main()
