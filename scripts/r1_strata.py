#!/usr/bin/env python3
"""Emit results/analysis/r1_cells.json -- the r1 (A31/A34) cells with IV/OOV
strata, base-weighted, in the same record shape as a36_cells.json.

  D34_DIR=<dir with d34*.result.parquet> A34_AP=<dir with ph_a34_*> \
  A31_DIR=<dir with {legs,ap}/> .venv/bin/python scripts/r1_strata.py

Sources: A34 d34 legs (natural, 186 image set, incl. the un-rephrased baseline
arm) and A31 b31/a30/c32 legs (adversarial + original + scaffold arms). The leg
parquets live on origin under results/rules_runs/r1_sim/jobs/; the env vars
point at local copies (they are fetched to a scratch dir, not the sparse tree).
Keys: "r1<diet>|<applier>|<cond>", "sc|<applier>|<cond>", "noreph|nat".
"""
import glob, json, os, pathlib, re, collections
import pandas as pd

R = pathlib.Path.home() / "dev/robotics/phrase-rl"
D34 = os.environ["D34_DIR"]; A34AP = os.environ["A34_AP"]; A31 = os.environ["A31_DIR"]

audit = json.load(open(R / "results/analysis/sealed_vocab_audit.json"))
def stem(n): return n.replace("put ", "").replace(" on ", "_on_").replace(" ", "_")
STRAT = {stem(x["nominal"]): ("iv" if x["stratum"] == "in-vocab" else "oov")
         for x in audit}
ALIAS = {"pepsi_on_plate": "pepsi_can_on_plate",
         "cube_on_plate": "green_cube_on_plate"}  # task-stem -> audit-stem
def strat(t):
    t = t.replace("widowx_", "").replace("_clean", "")
    s = STRAT.get(ALIAS.get(t, t))
    assert s is not None, f"unmapped task stratum: {t}"
    return s


def pool(leg_files):
    L = pd.concat([pd.read_parquet(f) for f in leg_files], ignore_index=True)
    return L.groupby(["task", "phrase"], as_index=False).agg(succ=("gt_success", "mean"))


def cellrec(leg_files, apply_pq, phrase_col="phrase"):
    L = pool(leg_files)
    a = pd.read_parquet(apply_pq)
    if phrase_col != "phrase":
        a = a.rename(columns={phrase_col: "phrase"})
    m = a.merge(L, on=["task", "phrase"], how="left")
    m["s"] = m.task.map(strat)
    return {"pooled": round(m.succ.mean(), 1),
            "iv": round(m[m.s == "iv"].succ.mean(), 1),
            "oov": round(m[m.s == "oov"].succ.mean(), 1),
            "n_base": int(m.succ.notna().sum()), "legs": len(leg_files)}


out = {}
AP = {"claude": "cl", "gemini": "ge", "qwen": "qw"}
# --- A34 natural (d34 legs, ph_a34 applies) ---
for bk, tag in [("s", "s"), ("b", "b"), ("t", "t"), ("sc", "sc")]:
    for ap, a2 in [("claude", "cln"), ("gemini", "gen"), ("qwen", "qwn")]:
        legs = glob.glob(f"{D34}/d34{tag}{a2}_*.result.parquet")
        pq = f"{A34AP}/ph_a34_{tag}_{ap}_nat.parquet"
        if legs and os.path.exists(pq):
            key = f"sc|{ap}|nat" if bk == "sc" else f"r1{bk}|{ap}|nat"
            out[key] = cellrec(legs, pq)
# un-rephrased natural baseline: the bases are their own "apply table"
base_legs = glob.glob(f"{D34}/d34basen_*.result.parquet")
if base_legs:
    L = pool(base_legs); L["s"] = L.task.map(strat)
    out["noreph|nat"] = {"pooled": round(L.succ.mean(), 1),
                         "iv": round(L[L.s == "iv"].succ.mean(), 1),
                         "oov": round(L[L.s == "oov"].succ.mean(), 1),
                         "n_base": len(L), "legs": len(base_legs)}
# --- A31 adversarial + original (b31 legs, ph_a31 applies) ---
for bk in ["t", "s", "b"]:
    for ap, a2 in AP.items():
        for c, cl in [("adv", "a"), ("orig", "o")]:
            legs = glob.glob(f"{A31}/legs/b31{bk}{a2}{cl}_*.result.parquet")
            pq = f"{A31}/ap/ph_a31_{bk}_{ap}_{c}.parquet"
            if legs and os.path.exists(pq):
                out[f"r1{bk}|{ap}|{c}"] = cellrec(legs, pq)
# --- A31 scaffold adversarial (a30 legs, ph_a29 applies) ---
for ap, a2 in AP.items():
    legs = glob.glob(f"{A31}/legs/a30{a2}adv_*.result.parquet")
    pq = f"{A31}/ap/ph_a29_{ap}sc_adv.parquet"
    if legs and os.path.exists(pq):
        out[f"sc|{ap}|adv"] = cellrec(legs, pq)

print(json.dumps(out, indent=1))
