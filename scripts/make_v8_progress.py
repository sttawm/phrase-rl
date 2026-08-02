#!/usr/bin/env python3
"""v8 progress — 2-panel: REAL val-8 rollouts (both conditions vs references) |
GRPO dynamics. v8 = tag-free cold start from base Qwen, input-dropout 0.5,
beta=0.15, grip-pure C=10 F=4. Curves from the pod6 all-in-one worker (stride 10,
2-rep greedy, tag-free gen). References: frozen Qwen (true step-0 of this
lineage), v7a-120 val-8 bests, and the sealed rules-v4 bars for context.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(*files):
    by = {}
    for f in files:
        if not os.path.exists(f):
            continue
        for l in open(f):
            l = l.strip()
            if not l or l.startswith(("<", "=", ">")):
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r["step"] not in by or r.get("n", 0) > by[r["step"]].get("n", 0):
                by[r["step"]] = r
    return sorted(by.values(), key=lambda d: d["step"])


adv = rows("results/analysis/v8_adv_curve.jsonl")
pol = rows("results/analysis/v8_pol_curve.jsonl")
V9OFF = 113  # v9 forked from v8@latest (step ~113); plotted as continuation
adv9 = rows("results/analysis/v9_adv_curve.jsonl")
pol9 = rows("results/analysis/v9_pol_curve.jsonl")
tr = [json.loads(l) for l in open("results/analysis/v8_train_log.jsonl")
      if '"kl"' in l]
tr = [d for d in tr if d.get("kl") is not None]
try:
    tr9 = [json.loads(l) for l in open("results/analysis/v9_train_log.jsonl") if '"kl"' in l]
    tr += [{**d, "step": V9OFF + d["step"]} for d in tr9 if d.get("kl") is not None]
except FileNotFoundError:
    pass

IV = {"widowx_carrot_on_plate", "widowx_spoon_on_towel", "widowx_stack_cube",
      "widowx_put_eggplant_in_basket"}


def strata(r):
    pt = r["per_task"]
    iv = [v for t, v in pt.items() if t in IV]
    oo = [v for t, v in pt.items() if t not in IV]
    return sum(iv) / len(iv), sum(oo) / len(oo)


fig, (a1, a1b, a2) = plt.subplots(1, 3, figsize=(18.5, 4.6))

xmax = max([r["step"] for r in adv + pol] + [V9OFF + r["step"] for r in adv9 + pol9] + [40]) + 8
for ax, rows_, title, refs in [
        (a1, pol, "POLISH (rewriting originals)",
         [(53.7, "originals IV 53.7", "#2f855a", "--"), (27.4, "originals OOV 27.4", "#2f855a", ":"),
          (58.6, "oracle IV", "#822727", "--"), (50.9, "oracle OOV", "#822727", ":")]),
        (a1b, adv, "REPAIR (rewriting adversarial)",
         [(41.3, "passthrough IV 41.3", "#718096", "--"), (27.7, "passthrough OOV 27.7", "#718096", ":"),
          (58.6, "oracle IV", "#822727", "--"), (50.9, "oracle OOV", "#822727", ":")])]:
    ax.set_xlim(-2, xmax)
    trx = ax.get_yaxis_transform()
    for val, lbl, col, ls in refs:
        ax.axhline(val, ls=ls, color=col, lw=1.1, alpha=0.75)
        ax.text(0.02, val + 0.35, lbl, color=col, fontsize=6.5, transform=trx)
    if rows_:
        xs = [r["step"] for r in rows_]
        ivs, oos = zip(*[strata(r) for r in rows_])
        ax.plot(xs, [r["pooled"] for r in rows_], marker="o", ms=4, color="#a0aec0",
                lw=1.0, ls="-", alpha=0.8, label="pooled")
        ax.plot(xs, ivs, marker="D", ms=6, color="#2b6cb0", lw=2, label="IN-VOCAB (4 tasks)")
        ax.plot(xs, oos, marker="s", ms=6, color="#dd6b20", lw=2, ls="--", label="OUT-OF-VOCAB (4 tasks)")
    r9 = adv9 if title.startswith("REPAIR") else pol9
    if r9:
        xs9 = [V9OFF + r["step"] for r in r9]
        iv9, oo9 = zip(*[strata(r) for r in r9])
        ax.plot(xs9, iv9, marker="D", ms=6, color="#6b46c1", lw=2, label="v9 IN-VOCAB (replay fork)")
        ax.plot(xs9, oo9, marker="s", ms=6, color="#c05621", lw=2, ls="--", label="v9 OOV")
        ax.plot(xs9, [r["pooled"] for r in r9], marker="o", ms=4, color="#718096", lw=1.0, alpha=0.8)
    ax.axvline(V9OFF, ls=":", color="#6b46c1", lw=1.0, alpha=0.7)
    ax.set_ylim(15, 62)
    ax.set_xlabel("v8 step")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)
a1.set_ylabel("val-8 rollout success % (stratum mean)")

if tr:
    ts = [d["step"] for d in tr]
    a2.plot(ts, [d["kl"] for d in tr], color="#805ad5", lw=1.4, label="KL(policy‖ref)")
    gn = [(d["step"], d["grad_norm"]) for d in tr if d.get("grad_norm") is not None]
    if gn:
        ag = a2.twinx()
        ag.plot([x for x, _ in gn], [g for _, g in gn], color="#dd6b20", lw=0.8,
                alpha=0.6, label="grad norm")
        ag.set_ylabel("grad norm", color="#dd6b20", fontsize=9)
        ag.tick_params(axis="y", labelcolor="#dd6b20", labelsize=7)
a2.set_xlabel("v8 step")
a2.set_ylabel("KL divergence", color="#805ad5")
a2.tick_params(axis="y", labelcolor="#805ad5")
a2.set_title("GRPO dynamics (beta=0.15, KL abort 1.2)")
a2.legend(loc="upper left", fontsize=8)
a2.grid(alpha=0.25)

fig.suptitle("v8 -> v9 RL — tag-free, grip-pure C=10 F=4, beta=0.15; v9 = replay fork @113 (16 ctx + 16 replayed, clip 0.2)   "
             "[sealed bars to beat: rules-v4 nominal 37.18 / ERT 33.30]", fontsize=10)
fig.tight_layout()
fig.savefig("results/charts/v8_progress.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/v8_progress.png")
