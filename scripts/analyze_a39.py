#!/usr/bin/env python3
"""A39 analysis: human naturals — raw baseline vs every apply arm.

Loads the b39roll leg results (24 layouts x 1 rep per phrase), attributes
episodes to arms through the apply mapping tables (frontier: local
ph_a39_<book>_<applier>.parquet; qwen: a39<book>qw job results), and reports
base-weighted success per arm with a paired-by-base sign-flip p vs the raw
human baseline. Register slices (adult/kid/robot) via the human table; a
phrase submitted under several registers counts in each (correct for
register-level means). Outputs results/analysis/a39_human_cells.json and
prints the summary.
"""
import glob
import itertools
import json
import pathlib

import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
jd = R / "results/rules_runs/r1_sim/jobs"
rng = np.random.default_rng(20260913)

legs = [pd.read_parquet(f) for f in glob.glob(str(jd / "b39roll_*.result.parquet"))]
L = pd.concat(legs, ignore_index=True)
rates = L.groupby(["task", "phrase"]).agg(succ=("gt_success", "mean"),
                                          nleg=("gt_success", "size"))
print(f"legs: {len(legs)}; rated phrases: {len(rates)}")

ph = pd.read_parquet(R / "results/human_naturals/a39_human_phrases.parquet")
raw = ph.drop_duplicates(["task", "phrase"]).merge(
    rates, on=["task", "phrase"], how="left")
missing_raw = raw[raw.succ.isna()]
if len(missing_raw):
    print(f"WARNING: {len(missing_raw)} raw bases unrated (legs pending)")
raw = raw.dropna(subset=["succ"])
base_rate = raw.set_index(["task", "phrase"]).succ
base_rate.index.names = ["task", "base"]

BOOKS = ["s", "b", "t", "s2", "s3", "b2", "b3", "t2", "t3", "sc"]


def arm_map(book, applier):
    if applier != "qwen":
        p = R / f"results/human_naturals/ph_a39_{book}_{applier}.parquet"
        return pd.read_parquet(p)[["task", "base", "phrase"]] if p.exists() else None
    hits = glob.glob(str(jd / f"a39{book}qw_*.result.parquet"))
    if not hits:
        return None
    d = pd.read_parquet(hits[0]).rename(columns={"phrase": "base", "rewrite": "phrase"})
    d["phrase"] = d.phrase.fillna("").astype(str)
    d.loc[d.phrase.str.strip() == "", "phrase"] = d.base
    return d[["task", "base", "phrase"]]


def sign_flip_p(d):
    d = np.asarray(d, float)
    if len(d) <= 12:
        signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    else:
        signs = rng.choice([1, -1], size=(20000, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float(((null >= abs(d.mean()) - 1e-12).sum() + (0 if len(d) <= 12 else 1))
                 / (len(null) + (0 if len(d) <= 12 else 1)))


reg_of = ph.groupby(["task", "phrase"]).register.apply(set)
OUT = {"raw_human": {"pooled": round(float(base_rate.mean()), 2),
                     "n_base": int(len(base_rate))}}
for regname in ["adult", "kid", "robot"]:
    keys = [k for k in base_rate.index if regname in reg_of.get(k, set())]
    OUT["raw_human"][regname] = round(float(base_rate.loc[keys].mean()), 2)

for book in BOOKS:
    for ap in ["claude", "gemini", "qwen"]:
        m = arm_map(book, ap)
        if m is None:
            continue
        j = m.merge(rates, on=["task", "phrase"], how="left")
        j = j.rename(columns={"succ": "c"}).set_index(["task", "base"])
        j = j.join(base_rate.rename("b"))
        j = j.dropna(subset=["c", "b"])
        d = (j.c - j.b).values
        rec = {"pooled": round(float(j.c.mean()), 2),
               "delta_vs_raw": round(float(d.mean()), 2),
               "p_perm": round(sign_flip_p(d), 4), "n_base": int(len(j))}
        for regname in ["adult", "kid", "robot"]:
            keys = [k for k in j.index if regname in reg_of.get(k, set())]
            sub = j.loc[keys]
            rec[regname] = round(float(sub.c.mean()), 2)
            rec[f"{regname}_delta"] = round(float((sub.c - sub.b).mean()), 2)
        OUT[f"{book}|{ap}"] = rec

out = R / "results/analysis/a39_human_cells.json"
out.write_text(json.dumps(OUT, indent=1))
print("cells ->", out)
rh = OUT["raw_human"]
print(f"\nraw human: {rh['pooled']}  (adult {rh['adult']} / kid {rh['kid']} / robot {rh['robot']})  n={rh['n_base']}")
print(f"{'arm':14s} {'pooled':>6} {'Δraw':>6} {'p':>7}  {'adult':>6} {'kid':>6} {'robot':>6}")
for k, v in OUT.items():
    if k == "raw_human":
        continue
    print(f"{k:14s} {v['pooled']:6.1f} {v['delta_vs_raw']:+6.1f} {v['p_perm']:7.4f}  "
          f"{v['adult']:6.1f} {v['kid']:6.1f} {v['robot']:6.1f}")
