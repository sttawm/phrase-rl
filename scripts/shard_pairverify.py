#!/usr/bin/env python3
"""Shard the remaining pair-verification work across pods.

Reads the manifest and cv2's partial results, computes remaining episodes per
(task, phrase) (grid x 3 reps minus recorded rows), and greedy-balances
phrases across the given pod names. Phrases with any progress pin to cv2 (its
--out file already holds their done keys, so resume is free); untouched
phrases are balanced by remaining-episode load. Emits
/tmp/pairshard_<pod>_{24,60}.parquet (task, arm, phrase).

  .venv/bin/python scripts/shard_pairverify.py /tmp/cv2_partial24.parquet [/tmp/cv2_partial60.parquet]
"""
import pathlib
import sys

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
PODS = ["cv2", "cv4", "cv5"]

U = pd.read_parquet(R / "results/analysis/bridge_pair_manifest.parquet")
U["arm"] = U.source
U["total_eps"] = U.grid * 3

done = pd.DataFrame(columns=["task", "phrase"])
parts = [pd.read_parquet(p) for p in sys.argv[1:] if pathlib.Path(p).exists()]
if parts:
    done = pd.concat(parts, ignore_index=True)
dc = done.groupby(["task", "phrase"]).size().rename("done").reset_index() if len(done) \
    else pd.DataFrame(columns=["task", "phrase", "done"])
U = U.merge(dc, on=["task", "phrase"], how="left").fillna({"done": 0})
U["remaining"] = (U.total_eps - U.done).clip(lower=0).astype(int)

loads = {p: 0 for p in PODS}
assign = {}
# progressed phrases stay on cv2
for r in U.itertuples():
    if r.done > 0:
        assign[(r.task, r.phrase)] = "cv2"
        loads["cv2"] += r.remaining
# untouched: greedy balance, largest first
rest = U[U.done == 0].sort_values("remaining", ascending=False)
for r in rest.itertuples():
    pod = min(loads, key=loads.get)
    assign[(r.task, r.phrase)] = pod
    loads[pod] += r.remaining

U["pod"] = U.apply(lambda r: assign[(r.task, r.phrase)], axis=1)
print("remaining episode load per pod:", loads)
print(U.groupby(["pod", "grid"]).size().rename("phrases").to_string())
for pod in PODS:
    for grid in [24, 60]:
        sub = U[(U.pod == pod) & (U.grid == grid) & (U.remaining > 0)][["task", "arm", "phrase"]]
        out = f"/tmp/pairshard_{pod}_{grid}.parquet"
        sub.to_parquet(out, index=False)
        if len(sub):
            print(f"{out}: {len(sub)} phrases")
