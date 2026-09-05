#!/usr/bin/env python3
"""A36 cells, BASE-WEIGHTED, split into IV / OOV strata (Fig-9 format).

  .venv/bin/python scripts/a36_cells_strata.py > results/analysis/a36_cells.json

Emits {tag|applier|cond: {pooled, iv, oov, n_base, n_dist, legs}} for every A36
arm whose legs are on disk. Pooling is over phrase BASES (join apply table to leg
results on task+phrase, mean over bases), the convention confirmed to match all
33 published A31 cells. Strata from results/analysis/sealed_vocab_audit.json
(5 in-vocab / 7 OOV).
"""
import glob, json, os, pathlib, re, collections
import pandas as pd

R = pathlib.Path.home() / "dev/robotics/phrase-rl"
JD = R / "results/rules_runs/r1_sim/jobs"
SD = R / "results/sealed"
AP = {"cl": "claude", "ge": "gemini", "qw": "qwen"}
CN = {"a": "adv", "n": "nat", "o": "orig"}

audit = json.load(open(R / "results/analysis/sealed_vocab_audit.json"))
# nominal "put carrot on ramekin" -> stem "carrot_on_ramekin"
def stem(nom):
    return nom.replace("put ", "").replace(" on ", "_on_").replace(" ", "_")
STRAT = {stem(x["nominal"]): ("iv" if x["stratum"] == "in-vocab" else "oov")
         for x in audit}
# leg task col is e.g. widowx_carrot_on_ramekin_clean; pepsi audit stem is
# "pepsi_can_on_plate" but the sealed stem is "pepsi_on_plate"
ALIAS = {"pepsi_on_plate": "pepsi_can_on_plate",
         "cube_on_plate": "green_cube_on_plate"}  # task-stem -> audit-stem


def task_stratum(task):
    t = task.replace("widowx_", "").replace("_clean", "")
    s = STRAT.get(ALIAS.get(t, t))
    assert s is not None, f"unmapped task stratum: {t}"
    return s


legs = collections.defaultdict(list)
for f in glob.glob(str(JD / "f36*.result.parquet")):
    m = re.match(r"f36([tsb][23])(cl|ge|qw)([ano])_", pathlib.Path(f).name)
    if m:
        legs[m.groups()].append(pd.read_parquet(f))

out = {}
for (tag, ap, c) in legs:
    L = (pd.concat(legs[(tag, ap, c)], ignore_index=True)
         .groupby(["task", "phrase"], as_index=False)
         .agg(succ=("gt_success", "mean"), neps=("n_ctx", "sum")))
    apf = SD / f"ph_a36_{tag}_{AP[ap]}_{CN[c]}.parquet"
    if not apf.exists():
        continue
    a = pd.read_parquet(apf).merge(L, on=["task", "phrase"], how="left")
    a["strat"] = a["task"].map(task_stratum)
    rec = {"pooled": round(a["succ"].mean(), 1),
           "iv": round(a[a.strat == "iv"]["succ"].mean(), 1),
           "oov": round(a[a.strat == "oov"]["succ"].mean(), 1),
           "n_base": int(a["succ"].notna().sum()),
           "n_dist": int(len(L)),
           "legs": len(legs[(tag, ap, c)])}
    out[f"{tag}|{AP[ap]}|{CN[c]}"] = rec

print(json.dumps(out, indent=1))
