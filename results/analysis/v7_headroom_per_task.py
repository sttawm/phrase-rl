"""v7 adjudication table: per-task headroom (frozen clean greedy vs frozen ERT
repair) and per-task tuned-vs-repair deltas across v6-B ladder checkpoints.

Sources: results/val_screens/eval12nom_frozen.parquet (x12 clean anchor),
screen_BASELINES_ert_x3.parquet (x3 frozen ERT repair + passthrough),
eval12_B_v6step_*.parquet (x12 tuned ladder verdicts).

Run: .venv/bin/python results/analysis/v7_headroom_per_task.py
Out: results/analysis/v7_headroom_per_task.json
"""

import glob
import json
import os
import re

import pandas as pd

os.chdir("/Users/sttawm/dev/robotics/phrase-rl")

b = pd.read_parquet("results/val_screens/screen_BASELINES_ert_x3.parquet")
per = b.groupby(["arm", "task"]).success.mean().unstack(0) * 100
frozen_clean = pd.read_parquet("results/val_screens/eval12nom_frozen.parquet") \
    .groupby("task").success.mean() * 100

t = pd.DataFrame({
    "frozen_clean_greedy_x12": frozen_clean,
    "frozen_ert_repair_x3": per["base_greedy"],
    "ert_passthrough_x3": per["redteam_direct"],
})
for f in sorted(glob.glob("results/val_screens/eval12_B_v6step_*.parquet")):
    s = int(re.search(r"v6step_(\d+)", f).group(1))
    t[f"v6B_s{s}_x12"] = pd.read_parquet(f).groupby("task").success.mean() * 100
t["headroom_clean_minus_repair"] = t.frozen_clean_greedy_x12 - t.frozen_ert_repair_x3
last = [c for c in t.columns if c.startswith("v6B_")][-1]
t["tuned_minus_repair_last"] = t[last] - t.frozen_ert_repair_x3

t = t.round(1)
out = {"per_task": t.to_dict(orient="index"),
       "pooled": t.mean().round(1).to_dict(),
       "note": ("repair/passthrough at x3 reps (n=72/task, +/-~9pp task SE); "
                "clean anchor and tuned at x12 (n=288/task). Several tasks show "
                "identical tuned numbers across checkpoints: greedy repair phrase "
                "is byte-identical across checkpoints and rollouts are CRN-pinned.")}
json.dump(out, open("results/analysis/v7_headroom_per_task.json", "w"), indent=1)
print(t.to_string())
print("\npooled:", out["pooled"])
