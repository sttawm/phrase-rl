#!/usr/bin/env python3
"""v11 RL progress.

Left:  proxy-val trajectory (v10 shown for contrast).
Right: ground truth on the nat24 probe — 24 distinct natural inputs per task,
       one per layout, so each cell rests on 192 phrases instead of 8. Both
       references sit on this same probe:
         passthrough = natural instruction straight to pi0, no rephraser
         step 0      = frozen Qwen + prompt B+ (the rules scaffold, untrained)
       so passthrough->step0 isolates the scaffold and step0->step N isolates RL.
The superseded 8-phrase cells are drawn faintly, for context only.
"""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = ("/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/"
           "a45839fa-48ee-4aea-bdee-72eb4fc9dccf/scratchpad")
v11 = [json.loads(l) for l in open(f"{SCRATCH}/v11_train_log.jsonl")]
v11v = sorted((r for r in v11 if r.get("type") == "val"), key=lambda r: r["step"])
v10 = [json.loads(l) for l in open("results/analysis/v10_telemetry/train_log.jsonl")]
v10v = sorted((r for r in v10 if r.get("type") == "val"), key=lambda r: r["step"])


def grab(prefix):
    out = {}
    for f in glob.glob(f"results/analysis/v11cells/{prefix}*.json"):
        c = json.load(open(f))
        key = str(c["step"])
        out[key.replace("nat24_", "").replace("nat_", "")] = c["pooled"]
    return out


new = grab("nat24_")
old = {k: v for k, v in grab("nat_").items()}
for k in list(old):
    if k in new and old[k] == new[k]:
        del old[k]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.8, 4.6))

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
a1.set_title("PROXY: v11 already past v10's whole-run best", fontsize=10)
a1.legend(fontsize=7.5, loc="lower right")
a1.grid(alpha=0.25)

for key, lbl, col in [("pass", "no rephraser (passthrough)  37.5", "#a0aec0"),
                      ("0000", "step 0: frozen Qwen + prompt B+  42.2", "#805ad5")]:
    if key in new:
        a2.axhline(new[key], ls="--", color=col, lw=1.5)
        a2.text(10, new[key] + 0.45, lbl, fontsize=8, color=col, fontweight="bold")
if "pass" in new and "0000" in new:
    a2.annotate("", xy=(99, new["0000"]), xytext=(99, new["pass"]),
                arrowprops=dict(arrowstyle="<->", color="#805ad5", lw=1.5))
    a2.text(96.5, (new["pass"] + new["0000"]) / 2, "rules\nscaffold\n+4.7",
            fontsize=8, color="#805ad5", ha="right", va="center", fontweight="bold")

pts = sorted((int(k), v) for k, v in new.items() if k.isdigit() and int(k) > 0)
if pts:
    a2.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#0d9488", lw=2.4, ms=9,
            label="RL checkpoints (nat24)")
    for x, y in pts:
        a2.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="#134e4a", fontweight="bold")
else:
    a2.text(52, 34.5, "checkpoint cells re-measuring on this probe\n(e4 + e8 backfilling steps 10–50)",
            fontsize=9, ha="center", color="#0d9488", style="italic",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f0fdfa", ec="#0d9488", lw=1.2))

opts = sorted((int(k), v) for k, v in old.items() if k.isdigit() and int(k) > 0)
if opts:
    a2.plot([p[0] for p in opts], [p[1] for p in opts], "o-", color="#cbd5e0", lw=1.3, ms=4,
            label="superseded 8-phrase probe", zorder=1)
a2.set_xlim(5, 105)
a2.set_ylim(30, 48)
a2.set_xlabel("v11 step")
a2.set_ylabel("val-8 success on natural inputs (%)")
a2.set_title("GROUND TRUTH: nat24 probe — 192 distinct phrases per cell", fontsize=10)
a2.legend(fontsize=7.5, loc="lower right")
a2.grid(alpha=0.25)

fig.suptitle("v11 — prompt B+ (Qwen mini-rules), 25/50/25 original/natural/adversarial, c4b reward",
             fontsize=10.5)
fig.tight_layout()
fig.savefig("results/charts/v11_progress.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/v11_progress.png")
