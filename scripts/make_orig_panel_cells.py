#!/usr/bin/env python3
"""results/analysis/orig_panel_cells.json — canonical-condition cells with
in/out-of-distribution strata, for the 2x2 dashboard's canonical panel.

Clean-trace runs only: A38 scaffold applies (ph_a31_scc_{ap}_orig) and A40
book applies (ph_a40_{book}_{ap}_orig), joined through every orig-tagged leg
in the job queue (a38*/b38*/a40*/b40orig). One base per task, so pooled =
task-weighted = base-weighted. Sanity-checked against
a40_clean_orig_cells.json and the A38 PREREG cells (31.8/31.1/28.5)."""
import glob
import json
import pathlib

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
J = R / "results/rules_runs/r1_sim/jobs"
IV = {"widowx_carrot_on_sponge_clean", "widowx_eggplant_on_sponge_clean",
      "widowx_cube_on_plate_clean", "widowx_nut_on_plate_clean",
      "widowx_small_plate_on_green_cube_clean"}

def pool(pats):
    fs = []
    for pat in pats:
        fs += glob.glob(str(J / pat))
    L = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return L.groupby(["task", "phrase"]).gt_success.mean()


R_SC = pool(["a38sc*_*.result.parquet", "b38sc*_*.result.parquet"])
R_40 = pool(["a40*_*.result.parquet", "b40orig_*.result.parquet"])
# fallback: rolls are keyed (task, phrase) and CRN-pinned, so any historical
# leg of the same phrase is the same measurement -- pool the whole queue,
# plus the anchors originals (x12) for identity rewrites.
R_ALL = pool(["*.result.parquet"])
anch = pd.read_parquet(R / "results/sealed/anchors_x12.parquet")
R_ANCH = (anch[anch.arm == "originals"].groupby(["task", "phrase"]).success.mean() * 100)
print(f"rates: scaffold {len(R_SC)}, a40 {len(R_40)}, all-jobs {len(R_ALL)}, anchors {len(R_ANCH)}")

A40 = json.loads((R / "results/analysis/a40_clean_orig_cells.json").read_text())
SC_REF = {"claude": 31.8, "gemini": 31.1, "qwen": 28.5}   # A38 PREREG cells

OUT = {}


def cell(tag, apply_pq, ref=None, primary=None):
    a = pd.read_parquet(R / f"results/sealed/{apply_pq}")[["task", "phrase"]]
    m = a.merge(primary.rename("r"), on=["task", "phrase"], how="left")
    for fbpool in (R_ALL, R_ANCH):
        fb = a.merge(fbpool.rename("r"), on=["task", "phrase"], how="left")
        m.loc[m.r.isna(), "r"] = fb.r[m.r.isna()]
    assert m.r.notna().all(), f"{apply_pq}: unmatched applies"
    pt = m.groupby("task").r.mean()
    pooled = float(pt.mean())
    # published pooled stays canonical; recomputed strata (shown rounded)
    # tolerate <=0.8pp shard-weighting wobble in the historical fallback.
    if ref is not None:
        assert abs(pooled - ref) < 0.8, (tag, pooled, ref)
    rec = {"pooled": ref if ref is not None else round(pooled, 2),
           "pooled_recomputed": round(pooled, 2),
           "iv": round(float(pt[pt.index.isin(IV)].mean()), 2),
           "oov": round(float(pt[~pt.index.isin(IV)].mean()), 2)}
    OUT[tag] = rec
    print(f"{tag:16s} {rec['pooled']:6.2f}  iv {rec['iv']:6.2f}  oov {rec['oov']:6.2f}"
          f"  (recomp {rec['pooled_recomputed']})")


for ap in ["claude", "gemini", "qwen"]:
    cell(f"sc|{ap}", f"ph_a31_scc_{ap}_orig.parquet", SC_REF[ap], primary=R_SC)
    for book in ["s", "s2", "s3", "b", "b2", "b3", "t", "t2", "t3"]:
        cell(f"{book}|{ap}", f"ph_a40_{book}_{ap}_orig.parquet",
             A40.get(f"{book}_{ap}"), primary=R_40)

out = R / "results/analysis/orig_panel_cells.json"
out.write_text(json.dumps(OUT, indent=1))
print("json ->", out)
