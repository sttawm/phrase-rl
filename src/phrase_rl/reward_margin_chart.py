"""Reward-over-time, both arms on one plot.

Rewards are in different units (flow residual vs normalized L2), so plot the
MARGIN over the original instruction as a percentage: 100*(orig_loss - cand_loss)
/ orig_loss, computed per step on the same contexts (paired, CRN). >0 = the
model's phrases beat the original instruction. Train = smoothed line, val = markers.

  .venv/bin/python -m phrase_rl.reward_margin_chart \
    --logs flow=data/probe_logs/flow.jsonl l2=data/probe_logs/l2.jsonl \
    --out results/charts/reward_margin.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"flow": "#D85A30", "l2": "#378ADD"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--out", default="results/charts/reward_margin.png")
    ap.add_argument("--smooth", type=int, default=25)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for arm, path in (kv.split("=", 1) for kv in args.logs):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        tr = [(r["step"], 100 * (r["orig_loss_mean"] - r["cand_loss_mean"]) / abs(r["orig_loss_mean"]))
              for r in rows if r.get("type") not in ("val", "probe")
              and r.get("cand_loss_mean") is not None and r.get("orig_loss_mean")]
        va = [(r["step"], 100 * (r["mean_reward"] - r["mean_orig_reward"]) / abs(r["mean_orig_reward"]))
              for r in rows if r.get("type") == "val" and r.get("mean_reward") is not None]
        steps, margins = zip(*tr)
        k = min(args.smooth, max(3, len(margins) // 5))
        smooth = np.convolve(margins, np.ones(k) / k, mode="valid")
        ax.plot(steps[k - 1:], smooth, color=COLORS.get(arm, "#888"), lw=2,
                label=f"{arm} — train margin (smoothed)")
        ax.plot(steps, margins, color=COLORS.get(arm, "#888"), alpha=0.15, lw=0.8)
        if va:
            vs, vm = zip(*va)
            ax.scatter(vs, vm, color=COLORS.get(arm, "#888"), s=90, marker="s",
                       edgecolor="black", zorder=5, label=f"{arm} — val margin")
    ax.axhline(0, color="#555", lw=1, ls="--")
    ax.set_ylim(-22, 22)  # raw per-step spikes reach -200%; the signal lives in +/-15
    ax.text(0.99, 0.03, "0 = original instruction (raw steps clipped at +/-22%)",
            transform=ax.transAxes, ha="right", fontsize=8, color="#555")
    ax.set_xlabel("training step")
    ax.set_ylabel("reward margin over original (%)\n(same contexts, paired CRN draws)")
    ax.set_title("Do the model's rephrases beat the original instruction? — flow vs L2 arm")
    ax.legend(fontsize=9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"chart -> {args.out}")


if __name__ == "__main__":
    main()
