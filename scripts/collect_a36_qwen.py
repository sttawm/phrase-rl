#!/usr/bin/env python3
"""Normalise A36 qwen pod-apply results into the ph_a36_* schema.

  FINAL_EVAL=1 .venv/bin/python scripts/collect_a36_qwen.py [tag[,tag...]]

Pod apply jobs return (task, phrase=BASE, rewrite=OUTPUT). The local appliers
write (task, [k,] base, phrase=OUTPUT). Rolling the pod schema directly would
send the UN-REPHRASED base to the simulator and look like a null result, so the
rename is load-bearing, not cosmetic.

Mirrors gen_a36_applies.py: blank rewrites fall back to the base, and the natural
condition is re-joined to the base table to recover `k` and deduped on (task, k).
"""
import glob, os, pathlib, re, sys
import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

ONLY = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
R = pathlib.Path.home() / "dev/robotics/phrase-rl"
jd = R / "results/rules_runs/r1_sim/jobs"
CN = {"a": "adv", "n": "nat", "o": "orig"}

nat_bases = pd.read_parquet(R / "results/sealed/ph_sealed_natural_v2_img.parquet")[
    ["task", "k", "phrase"]]

n = 0
for f in sorted(glob.glob(str(jd / "g36*qw*.result.parquet"))):
    m = re.match(r"g36([tsb][23])qw([ano])_", pathlib.Path(f).name)
    if not m:
        continue
    tag, c = m.groups()
    if ONLY and tag not in ONLY:
        continue
    cond = CN[c]
    rw = pd.read_parquet(f)
    if "rewrite" not in rw.columns:
        print(f"  SKIP {f}: no rewrite column, got {list(rw.columns)}")
        continue
    rw = rw.rename(columns={"phrase": "base", "rewrite": "phrase"})
    rw["phrase"] = rw.phrase.fillna("").astype(str)
    blank = rw.phrase.str.strip() == ""
    rw.loc[blank, "phrase"] = rw.loc[blank, "base"]
    if cond == "nat":
        rw = (nat_bases.rename(columns={"phrase": "base"})
              .merge(rw, on=["task", "base"], how="left")
              .drop_duplicates(["task", "k"]))
        rw["phrase"] = rw.phrase.fillna(rw.base)
    out = R / f"results/sealed/ph_a36_{tag}_qwen_{cond}.parquet"
    rw.to_parquet(out, index=False)
    changed = (rw.phrase.str.strip().str.lower()
               != rw.base.str.strip().str.lower()).mean()
    print(f"  [{tag}/qwen/{cond}] {len(rw)} rows, blank->base {int(blank.sum())}, "
          f"changed {changed:.0%} -> {out.name}")
    n += 1
print(f"collected {n} qwen apply files")
