#!/usr/bin/env python3
"""B41: the NON-augmented Bridge checkpoint (juexzz/INTACT-pi0-finetune-bridge)
on the 363 human-natural phrases, 24 layouts x 1 rep -- paired against the
rephrase-finetuned checkpoint's raw-human cell from A39 (same phrases, same
24 layouts, same CRN seeds). Writes results/analysis/b41_base_cells.json.

  .venv/bin/python scripts/analyze_b41.py
"""
import glob
import itertools
import json
import pathlib

import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
jd = R / "results/rules_runs/r1_sim/jobs"
rng = np.random.default_rng(20261001)


def rates_for(prefix):
    legs = [pd.read_parquet(f) for f in glob.glob(str(jd / f"{prefix}_*.result.parquet"))]
    L = pd.concat(legs, ignore_index=True)
    r = L.groupby(["task", "phrase"]).agg(succ=("gt_success", "mean"),
                                          n_ctx=("n_ctx", "sum"))
    return r, len(legs)


def sign_flip_p(d):
    d = np.asarray(d, float)
    if len(d) <= 12:
        signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    else:
        signs = rng.choice([1, -1], size=(20000, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float(((null >= abs(d.mean()) - 1e-12).sum() + (0 if len(d) <= 12 else 1))
                 / (len(null) + (0 if len(d) <= 12 else 1)))


ph = pd.read_parquet(R / "results/human_naturals/a39_human_phrases.parquet")
uniq = ph.drop_duplicates(["task", "phrase"])[["task", "phrase"]]
reg_of = ph.groupby(["task", "phrase"]).register.apply(set)

base, nb = rates_for("b41base")      # non-augmented checkpoint (this run)
aug, na = rates_for("b39roll")       # rephrase-finetuned checkpoint (A39)
j = uniq.merge(base.rename(columns={"succ": "base"}), on=["task", "phrase"], how="left")
j = j.merge(aug[["succ"]].rename(columns={"succ": "aug"}), on=["task", "phrase"], how="left")
miss = j.base.isna().sum()
if miss:
    print(f"WARNING: {miss} of {len(j)} human phrases unrated by the base checkpoint (legs pending)")
j = j.dropna(subset=["base", "aug"])
print(f"base legs {nb}, aug legs {na}; paired phrases {len(j)} over {j.task.nunique()} tasks")

d = j.aug - j.base
OUT = {"base_raw_human": {"pooled": round(float(j.base.mean()), 2), "n_base": int(len(j))},
       "aug_raw_human": {"pooled": round(float(j.aug.mean()), 2), "n_base": int(len(j))},
       "aug_minus_base": {"delta": round(float(d.mean()), 2), "p_perm": round(sign_flip_p(d), 4),
                          "n_base": int(len(j))}}
for regname in ["adult", "kid", "robot"]:
    m = [regname in reg_of.get((t, p), set()) for t, p in zip(j.task, j.phrase)]
    OUT["base_raw_human"][regname] = round(float(j.base[m].mean()), 2)
    OUT["aug_raw_human"][regname] = round(float(j.aug[m].mean()), 2)
# in-distribution (5 tasks) / out-of-distribution (7) strata, base-weighted
# (mean over phrases within the stratum) -- same split as make_pi0_conditions.py
IV = {"widowx_carrot_on_sponge_clean", "widowx_eggplant_on_sponge_clean",
      "widowx_cube_on_plate_clean", "widowx_nut_on_plate_clean",
      "widowx_small_plate_on_green_cube_clean"}
iv = j.task.isin(IV)
for col, key in (("base", "base_raw_human"), ("aug", "aug_raw_human")):
    OUT[key]["iv"] = round(float(j[col][iv].mean()), 2)
    OUT[key]["oov"] = round(float(j[col][~iv].mean()), 2)
OUT["aug_minus_base"]["iv_delta"] = round(float(d[iv].mean()), 2)
OUT["aug_minus_base"]["oov_delta"] = round(float(d[~iv].mean()), 2)
OUT["aug_minus_base"]["iv_p_perm"] = round(sign_flip_p(d[iv]), 4)
OUT["aug_minus_base"]["oov_p_perm"] = round(sign_flip_p(d[~iv]), 4)
a39 = json.load(open(R / "results/analysis/a39_human_cells.json"))["raw_human"]
assert abs(OUT["aug_raw_human"]["iv"] - a39["iv"]) < 0.02 and abs(OUT["aug_raw_human"]["oov"] - a39["oov"]) < 0.02, \
    (OUT["aug_raw_human"], a39)
per_task = (j.groupby("task").agg(base=("base", "mean"), aug=("aug", "mean"), n=("base", "size"))
             .round(2))
per_task["delta"] = (per_task.aug - per_task.base).round(2)
OUT["per_task"] = {t: {k: (int(v) if k == "n" else float(v)) for k, v in row.items()}
                   for t, row in per_task.to_dict("index").items()}
out = R / "results/analysis/b41_base_cells.json"
json.dump(OUT, open(out, "w"), indent=1)
print(f"\nbase (non-augmented) raw human: {OUT['base_raw_human']['pooled']}   "
      f"augmented raw human: {OUT['aug_raw_human']['pooled']}   "
      f"delta {OUT['aug_minus_base']['delta']:+.2f} (p={OUT['aug_minus_base']['p_perm']})")
b, a, m = OUT["base_raw_human"], OUT["aug_raw_human"], OUT["aug_minus_base"]
print(f"in-distribution:  base {b['iv']} aug {a['iv']}  delta {m['iv_delta']:+.2f} (p={m['iv_p_perm']})")
print(f"out-of-distrib.:  base {b['oov']} aug {a['oov']}  delta {m['oov_delta']:+.2f} (p={m['oov_p_perm']})")
print(per_task.to_string())
print(f"-> {out.relative_to(R)}")
