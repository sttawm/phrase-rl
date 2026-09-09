#!/usr/bin/env python3
"""A37 Anchor B verdict: CoVer-harness verifier-off vs our phase0c passthrough.

Inputs: anchorB_k1.jsonl / anchorB_k2.jsonl fetched from cv2 (their harness,
24 pinned layouts x 1 rep per wave), and the reference cells from our a29pass
legs (24 layouts x 2 reps) restricted to the same k1/k2 attacks.

  .venv/bin/python scripts/a37_anchorB_verdict.py /tmp/anchorB_k1.jsonl /tmp/anchorB_k2.jsonl

Pass criterion (A37): pooled agreement within CRN noise. Layouts are shared;
decode noise is re-drawn (their single-sample path vs ours), so per-cell
binomial jitter at n=24 is ~9pp; pooled over 24 cells the SE is ~2pp. We call
PASS at |delta| <= 4pp pooled, and report per-cell agreement for the record.
"""
import json
import pathlib
import sys

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]

rows = []
for p in sys.argv[1:]:
    with open(p) as f:
        rows += [json.loads(l) for l in f]
th = pd.DataFrame(rows)
th["succ"] = th.success.astype(float) * 100

erts = pd.read_parquet(R / "results/sealed/a29_new_erts.parquet")
erts["k"] = erts.groupby("task", sort=False).cumcount() + 1
th = th.merge(erts.rename(columns={"phrase": "base"})[["task", "base", "k"]],
              on=["task", "base"], how="left")
assert th.k.notna().all(), "episode base not found in a29_new_erts"

ref = pd.read_parquet("/tmp/anchorB_ref.parquet")

cell = th.groupby(["task", "k"], as_index=False).agg(cover=("succ", "mean"),
                                                     n=("succ", "size"))
cmp = cell.merge(ref.rename(columns={"gt_success": "ours"}), on=["task", "k"])
cmp["delta"] = cmp.cover - cmp.ours

pooled_c, pooled_o = cmp.cover.mean(), cmp.ours.mean()
print(cmp.sort_values(["task", "k"])[["task", "k", "ours", "cover", "delta", "n"]]
      .to_string(index=False))
print(f"\npooled: theirs-harness {pooled_c:.1f} vs ours {pooled_o:.1f} "
      f"(delta {pooled_c - pooled_o:+.1f}pp; episodes {int(cell.n.sum())})")
print("VERDICT:", "PASS" if abs(pooled_c - pooled_o) <= 4.0 else "FAIL",
      "(|pooled delta| <= 4pp)")
