#!/usr/bin/env python3
"""v11 RL progress — proxy-val trajectory (with v10 for contrast) + ground-truth
natural-probe cells. v11 = prompt B+ (QWEN_MINI_RULES), 25/50/25 input mix, c4b."""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = ("/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/"
           "a45839fa-48ee-4aea-bdee-72eb4fc9dccf/scratchpad")
v11 = [json.loads(l) for l in open(f"{SCRATCH}/v11_train_log.jsonl")]
v11v = sorted((r for r in v11 if r.get("type") == "val"), key=lambda r: r["step"])
v11k = sorted((r for r in v11 if r.get("kl") is not None), key=lambda r: r["step"])
v10 = [json.loads(l) for l in open("results/analysis/v10_telemetry/train_log.jsonl")]
v10v = sorted((r for r in v10 if r.get("type") == "val"), key=lambda r: r["step"])
cells = [json.load(open(f)) for f in glob.glob("results/analysis/v11cells/*.json")]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 4.5))

# --- panel 1: proxy val, v11 vs v10
a1.axhline(v11v[0]["mean_orig_reward"], ls=":", color="#718096", lw=1.2)
a1.text(150, v11v[0]["mean_orig_reward"] + 0.06, "originals (human phrasing) +0.62",
        fontsize=7.5, color="#718096", ha="center")
a1.plot([r["step"] for r in v10v], [r["mean_greedy_reward"] for r in v10v],
        "o-", color="#cbd5e0", lw=1.6, ms=4, label="v10 greedy (prompt B, 256 steps)")
a1.plot([r["step"] for r in v11v], [r["mean_greedy_reward"] for r in v11v],
        "o-", color="#0d9488", lw=2.4, ms=7, label="v11 greedy (prompt B+)")
a1.plot([r["step"] for r in v11v], [r.get("mean_best_reward") for r in v11v],
        "s--", color="#48bb78", lw=1.3, ms=4.5, label="v11 best-of-16")
a1.set_xlabel("training step")
a1.set_ylabel("proxy reward")
a1.set_title("Proxy val: v11 is already past v10's whole-run best\n"
             f"(v11 step {v11v[-1]['step']}: {v11v[-1]['mean_greedy_reward']:.2f} vs v10 best −0.67)",
             fontsize=9.5)
a1.legend(fontsize=7.5, loc="lower right")
a1.grid(alpha=0.25)

# --- panel 2: ground-truth natural-probe cells
# same-probe baselines only (sealed-suite numbers are a different task set)
base = {c["step"]: c["pooled"] for c in cells if not str(c["step"]).isdigit()}
for key, lbl, col in [("nat_pass", "no rephraser (passthrough)", "#a0aec0"),
                      ("nat_0000", "step 0: base Qwen + prompt B+", "#805ad5")]:
    if key in base:
        a2.axhline(base[key], ls="--", color=col, lw=1.3)
        a2.text(12, base[key] + 0.35, f"{lbl} ({base[key]:.1f})", fontsize=7.5, color=col)
pts = sorted((c for c in cells if str(c["step"]).isdigit()), key=lambda c: int(c["step"]))
xs = [int(c["step"]) for c in pts]
ys = [c["pooled"] for c in pts]
a2.plot(xs, ys, "o-", color="#0d9488", lw=2.2, ms=8)
for x, y in zip(xs, ys):
    a2.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8, color="#134e4a", fontweight="bold")
a2.set_ylim(26, 46)
a2.set_xlim(5, max(60, (max(xs) + 10) if xs else 60))
a2.set_xlabel("v11 step")
a2.set_ylabel("val-8 success on natural inputs (%)")
a2.set_title(f"GROUND TRUTH: natural-input probe ({len(xs)} checkpoint cells, 192 eps each)\n"
             "all references measured on this same probe", fontsize=9.5)
a2.grid(alpha=0.25)

fig.suptitle("v11 — prompt B+ (Qwen mini-rules), 25/50/25 original/natural/adversarial, c4b reward",
             fontsize=10.5)
fig.tight_layout()
fig.savefig("results/charts/v11_progress.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/v11_progress.png")
