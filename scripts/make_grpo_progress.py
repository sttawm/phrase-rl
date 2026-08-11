#!/usr/bin/env python3
"""results/charts/grpo_progress.png -- paper figure: GRPO reliably climbs its
proxy reward (left) while rollout success stays flat (right).

Left: greedy-decode proxy reward per training step -- v11 from the committed
telemetry (which ends at step ~80), v10 across its full 260 steps.
Right: nat24 ground-truth probes for v11, steps 0-240.
"""
import glob
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

v11 = [json.loads(l) for l in open("results/analysis/v11_telemetry/train_log.jsonl")]
v10 = [json.loads(l) for l in open("results/analysis/v10_telemetry/train_log.jsonl")]
v11v = [r for r in v11 if "mean_greedy_reward" in r]
v10v = [r for r in v10 if "mean_greedy_reward" in r]

cells = []
for f in sorted(glob.glob("results/analysis/v11cells/nat24_0*.json")):
    d = json.load(open(f))
    cells.append((int(re.search(r"nat24_(\d+)", f).group(1)), d["pooled"]))
cells.sort()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 4.4))

a1.plot([r["step"] for r in v10v], [r["mean_greedy_reward"] for r in v10v],
        "o-", color="#a0aec0", lw=1.6, ms=4, label="sibling run (v10), 260 steps")
a1.plot([r["step"] for r in v11v], [r["mean_greedy_reward"] for r in v11v],
        "o-", color="#0d9488", lw=2.4, ms=6, label="final run (v11), telemetry to step 80")
a1.set_xlabel("training step")
a1.set_ylabel("proxy reward (greedy decode)")
a1.set_title("The policy climbs its reward", fontsize=12)
a1.legend(fontsize=9, loc="lower right")
a1.grid(alpha=0.25)

a2.plot([c[0] for c in cells], [c[1] for c in cells], "o-", color="#0d9488",
        lw=2.2, ms=6, label="final run (v11) rewrites")
a2.axhline(42.19, ls="--", color="#805ad5", lw=1.4)
a2.text(238, 42.6, "step-0 policy 42.2", color="#805ad5", fontsize=9, ha="right")
a2.axhline(37.5, ls="--", color="#718096", lw=1.4)
a2.text(238, 36.5, "no rephraser 37.5", color="#718096", fontsize=9, ha="right")
a2.set_ylim(30, 48)
a2.set_xlabel("training step")
a2.set_ylabel("rollout success on natural probes (%)")
a2.set_title("Ground truth does not move", fontsize=12)
a2.legend(fontsize=9, loc="lower left")
a2.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("results/charts/grpo_progress.png", dpi=140, bbox_inches="tight",
            pad_inches=0.2)
print("chart -> results/charts/grpo_progress.png")
