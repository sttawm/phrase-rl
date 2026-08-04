#!/usr/bin/env python3
"""v9 reward telemetry — 2 panels: proxy-reward val (every 21 steps) | per-step
candidate grip error by input tier (the trainer's own running signal, grip-pure
C=10 F=4 with replay). v9b cutover (accum 6->1) marked at step 18."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

recs = [json.loads(l) for l in open("results/analysis/v9_train_log.jsonl")]
vals = sorted((d for d in recs if d.get("type") == "val"), key=lambda d: d["step"])
tr = sorted((d for d in recs if d.get("kl") is not None), key=lambda d: d["step"])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.4))

if vals:
    s = [v["step"] for v in vals]
    for key, label, kw in [
        ("mean_orig_reward", "orig (no rewrite)", {"color": "#718096", "ls": ":"}),
        ("mean_best_reward", "best-of-16", {"color": "#48bb78"}),
        ("mean_greedy_reward", "greedy (deploy)", {"color": "#2b6cb0"}),
        ("mean_reward", "sample mean", {"color": "#a0aec0", "ls": "--"}),
    ]:
        if any(v.get(key) is not None for v in vals):
            a1.plot(s, [v.get(key) for v in vals], marker="o", ms=5, label=label, **kw)
    a1.legend(fontsize=8, loc="center right")
a1.axvline(18, ls=":", color="#6b46c1", lw=1.2)
a1.text(18.4, a1.get_ylim()[0] + 0.1, "v9b (accum 1)", color="#6b46c1", fontsize=7, rotation=90)
a1.set_xlabel("v9 step")
a1.set_ylabel("proxy reward (calibrated logit scale)")
a1.set_title("proxy-reward val (every 21 steps) — candidates sit FAR below orig; no movement")
a1.grid(alpha=0.25)

if tr:
    ts = [d["step"] for d in tr]
    for tier, col in [("nominal", "#2f855a"), ("benign", "#2b6cb0"), ("ert", "#e53e3e")]:
        ys = [d.get("grip_by_tier", {}).get(tier) for d in tr]
        a2.plot(ts, ys, color=col, lw=1.0, alpha=0.6, label=f"{tier} tier")
    W = {"ert": 0.5, "nominal": 0.25, "benign": 0.25}
    wavg = []
    for d in tr:
        g = d.get("grip_by_tier", {})
        wavg.append(sum(W[t] * g[t] for t in W) if all(t in g for t in W) else None)
    a2.plot(ts, wavg, color="#1a202c", lw=2.2,
            label="TIER-WEIGHTED MEAN (0.5 ert / 0.25 nom / 0.25 benign)")
    w = [v for v in wavg if v is not None]
    if len(w) >= 10:
        early, late = sum(w[:5]) / 5, sum(w[-5:]) / 5
        a2.set_title(f"per-step candidate grip error by tier — weighted mean {early:.3f} -> {late:.3f}")
a2.axvline(18, ls=":", color="#6b46c1", lw=1.2)
a2.set_xlabel("v9 step")
a2.set_ylabel("candidate grip error (lower = better)")
a2.legend(fontsize=8)
a2.grid(alpha=0.25)

fig.suptitle("v9 reward telemetry — grip-pure C=10 F=4, replay 16 groups, tag-free; v9b@18 = one 512-cand update/step",
             fontsize=10.5)
fig.tight_layout()
fig.savefig("results/charts/v9_reward.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/v9_reward.png")
