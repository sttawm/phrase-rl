#!/usr/bin/env python3
"""v7a (beta=0.15) vs v7b (beta=0.05) from the fork point (v7 step 140).
Both trajectories on a shared 'steps since fork' axis so the post-fork
divergence is isolated. Left: KL (the experimental lever). Right: real rollout
(greedy solid, sampled dashed). v7b_step is offset by +FORK to align.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FORK = 140


def kl_series(path):
    out = []
    for l in open(path):
        d = json.loads(l)
        if d.get("type") not in ("val", "probe") and d.get("kl") is not None:
            out.append((d["step"], d["kl"]))
    return sorted(out)


def curve(path):
    rows = sorted((json.loads(l) for l in open(path)), key=lambda r: r["step"])
    return rows


v7a_kl = [(s, k) for s, k in kl_series("/tmp/v7_train_log.jsonl") if s >= FORK]
v7b_kl = [(FORK + s, k) for s, k in kl_series("/tmp/v7b_train_log.jsonl")]
v7a_roll = [r for r in curve("results/analysis/v7_curve_backfill.jsonl") if r["step"] >= FORK]
v7b_roll = curve("results/analysis/v7b_rollout_curve.jsonl")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))

a1.axvline(FORK, ls=":", color="#a0aec0", lw=1)
a1.text(FORK + 2, 0.16, "fork", color="#718096", fontsize=8)
a1.plot([s for s, _ in v7a_kl], [k for _, k in v7a_kl], color="#2b6cb0", lw=1.3, label="v7a  β=0.15")
a1.plot([s for s, _ in v7b_kl], [k for _, k in v7b_kl], color="#dd6b20", lw=1.6, label="v7b  β=0.05")
a1.set_xlabel("training step (aligned at fork = 140)")
a1.set_ylabel("KL(policy ‖ base)")
a1.set_title("KL divergence — the lever")
a1.legend(fontsize=9)
a1.grid(alpha=0.25)

a2.axvline(FORK, ls=":", color="#a0aec0", lw=1)
a2.plot([r["step"] for r in v7a_roll], [r["pooled"] for r in v7a_roll],
        color="#2b6cb0", lw=1.8, marker="o", ms=3, label="v7a greedy")
a2.plot([r["step"] for r in v7a_roll], [r["sampled_pooled"] for r in v7a_roll],
        color="#2b6cb0", lw=1.1, ls="--", alpha=0.7, marker=".", label="v7a sampled")
a2.plot([FORK + r["step"] for r in v7b_roll], [r["pooled"] for r in v7b_roll],
        color="#dd6b20", lw=1.8, marker="D", ms=4, label="v7b greedy")
a2.plot([FORK + r["step"] for r in v7b_roll], [r.get("sampled_pooled") for r in v7b_roll],
        color="#dd6b20", lw=1.1, ls="--", alpha=0.7, marker=".", label="v7b sampled")
a2.axhline(58.3, ls=":", color="#48bb78", lw=1)
a2.text(FORK + 4, 58.6, "v7 peak 58.3", color="#2f855a", fontsize=7)
a2.set_xlabel("training step (aligned at fork = 140)")
a2.set_ylabel("rollout success %")
a2.set_title("real rollout (n=192/pt)")
a2.legend(fontsize=7, ncol=2)
a2.grid(alpha=0.25)

fig.suptitle("v7a (β=0.15) vs v7b (β=0.05) — beta-sweep fork from v7 step 140", fontsize=11)
fig.tight_layout()
fig.savefig("results/charts/v7ab_compare.png", dpi=145, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/v7ab_compare.png")
print(f"v7a KL@end {v7a_kl[-1][1]:.3f}  v7b KL@end {v7b_kl[-1][1]:.3f}")
print(f"v7b rollout: {[(r['step'], r['pooled'], r.get('sampled_pooled')) for r in v7b_roll]}")
