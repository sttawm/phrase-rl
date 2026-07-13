"""Deployment chart: tuned checkpoints vs baselines, 95% CIs, step-0 anchored.

Key design point (user insight 2026-07-13): base greedy IS the step-0 point of
every tuned curve (LoRA zero-init + identical prompts/eval), so tuned curves are
drawn FROM the base point at step 0.

  .venv/bin/python -m phrase_rl.deploy_chart
"""

import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def stats(df, arm):
    d = df[df.arm == arm]
    if not len(d):
        return None
    p, n = d.success.mean(), len(d)
    return 100 * p, 196 * np.sqrt(p * (1 - p) / n), n


def main():
    b = pd.read_parquet("results/overnight/raw/rollouts_rt_baselines_crn.parquet")
    base = stats(b, "base")
    rand = stats(b, "base_random")
    old = pd.read_parquet("results/overnight/raw/rollouts_rt_baselines.parquet")
    direct = stats(old, "redteam_direct")

    pts = {"tuned_flow": {0: base}, "tuned_l2": {0: base}}
    for f in glob.glob("results/overnight/raw/rollouts_rt_f*_l*.parquet"):
        tag = f.split("rollouts_rt_")[1].split(".parquet")[0]
        fs, ls = tag.split("_")
        d = pd.read_parquet(f)
        for arm, step in [("tuned_flow", int(fs[1:])), ("tuned_l2", int(ls[1:]))]:
            s = stats(d, arm)
            if s and (step not in pts[arm] or s[2] > pts[arm][step][2]):
                pts[arm][step] = s
    late = stats(pd.read_parquet("results/overnight/raw/rollouts_l2_late.parquet"), "tuned_l2_late")
    if late:
        pts["tuned_l2"][620] = late
    # v3-only view: flow points beyond its run (v2 f300/f680) get dashed-ghost treatment
    V2_FLOW = {300, 680}

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for m, se, n, lab, c in [(base[0], base[1], base[2], "base greedy", "#333333"),
                             (rand[0], rand[1], rand[2], "base random", "#8B7BD8"),
                             (direct[0], direct[1], direct[2], "red-team direct", "#B4B2A9")]:
        ax.axhspan(m - se, m + se, color=c, alpha=0.10)
        ax.axhline(m, color=c, lw=1.4, ls="--")
        ax.text(1.005, m, f"{lab} {m:.1f}±{se:.1f} (n={n})", transform=ax.get_yaxis_transform(),
                fontsize=8, color=c, va="center")
    for arm, c in [("tuned_flow", "#D85A30"), ("tuned_l2", "#378ADD")]:
        d = pts[arm]
        v3 = sorted(s for s in d if s not in V2_FLOW or arm != "tuned_flow")
        xs = v3
        ms = [d[s][0] for s in xs]
        es = [d[s][1] for s in xs]
        ax.errorbar(xs, ms, yerr=es, marker="o", ms=6.5, capsize=4, lw=1.8, color=c,
                    label=f"{arm} (v3, from scratch; step 0 = base)")
        if arm == "tuned_flow":
            g = sorted(V2_FLOW & set(d))
            if g:
                ax.errorbar(g, [d[s][0] for s in g], yerr=[d[s][1] for s in g], marker="s",
                            ms=6, capsize=3, lw=1.2, ls=":", color=c, alpha=0.55,
                            label="tuned_flow (v2 arm, historical)")
    ax.set_xlabel("training step of the evaluated checkpoint")
    ax.set_ylabel("red-team success (%)\nCoVer ERT instructions, 24 layouts, CRN reps")
    ax.set_title("Proxy-reward deployment — final precision (x12–x24 reps per point)")
    ax.legend(fontsize=8.5, loc="lower right")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/charts/deploy_final_precision.png", dpi=130, bbox_inches="tight")
    print("chart -> results/charts/deploy_final_precision.png")
    for arm in pts:
        for s in sorted(pts[arm]):
            m, se, n = pts[arm][s]
            print(f"  {arm}@{s}: {m:.1f} ±{se:.1f} (n={n})")


if __name__ == "__main__":
    main()
