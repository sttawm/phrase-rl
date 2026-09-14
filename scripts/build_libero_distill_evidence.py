#!/usr/bin/env python3
"""Flat evidence file for LIBERO rule distillation: task, category, phrase, success, n.

  .venv/bin/python scripts/build_libero_distill_evidence.py

One row per phrase that takes part in at least one single-edit pair
(results/analysis/pi05_bank/single_edit_pairs.parquet), every phrase at n=50 on
inits 0-49. `category` is the edit dimension the phrase varies on: "canonical"
for the task's finetune string, otherwise the class of its pair with the
canonical, or (ladder rungs with no canonical pair) the commonest class among
its pairs. Tasks in the sealed set v2 (sealed_v2_draw.json, both halves) are
removed entirely -- no leakage -- and the script refuses to write while the
out-of-finetune half is still incomplete unless --provisional is given.
Writes results/analysis/pi05_bank/distill_evidence_v2.csv (+ .md rendering).
"""
import json
import pathlib
import sys

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
B = R / "results/analysis/pi05_bank"
LABEL = {"case": "capitalisation", "punctuation": "punctuation", "determiner": "determiner", "color": "colour word",
         "preposition/particle": "preposition or particle", "verb/frame": "verb or sentence frame",
         "words added": "words added", "words removed": "words removed", "noun/other": "object or place noun"}

draw = json.load(open(B / "sealed_v2_draw.json"))
sealed = set()
for name, h in draw["halves"].items():
    sealed |= set(h["accepted"])
    if not h["complete"] and "--provisional" not in sys.argv:
        raise SystemExit(f"{name} half incomplete ({len(h['accepted'])}/{draw['n_per_half']}); pass --provisional to write anyway")

P = pd.read_parquet(B / "single_edit_pairs.parquet")
P = P[~P.task.isin(sealed)]
rows = {}
for r in P.itertuples():
    for side in ("a", "b"):
        ph, succ = getattr(r, f"phrase_{side}"), getattr(r, f"succ_{side}")
        key = (r.task, ph)
        d = rows.setdefault(key, {"task": r.task, "phrase": ph, "success": int(succ), "n": 50, "cats": [], "canon_cat": None,
                                  "is_canon": ph == r.canonical})
        if ph == r.canonical:
            continue
        cat = LABEL[r.edit_class]
        d["cats"].append(cat)
        if r.a_is_canonical and side == "b":
            d["canon_cat"] = cat
out = []
for d in rows.values():
    if d["is_canon"]:
        cat = "canonical"
    else:
        cat = d["canon_cat"] or max(set(d["cats"]), key=d["cats"].count)
    out.append({"task": d["task"], "category": cat, "phrase": d["phrase"], "success": d["success"], "n": d["n"]})
E = pd.DataFrame(out)
E["_c"] = (E.category != "canonical").astype(int)
E = E.sort_values(["task", "_c", "category", "success"], ascending=[True, True, True, False]).drop(columns="_c")
E.to_csv(B / "distill_evidence_v2.csv", index=False)
md = ["| task | category | phrase | success % | n |", "|---|---|---|---|---|"]
md += [f"| {r.task} | {r.category} | {r.phrase} | {r.success} | {r.n} |" for r in E.itertuples()]
(B / "distill_evidence_v2.md").write_text("\n".join(md) + "\n")
print(f"sealed tasks removed: {len(sealed)} | rows: {len(E)} phrases on {E.task.nunique()} tasks "
      f"(in-finetune {E[E.task.str.startswith(('libero_spatial','libero_object','libero_goal','libero_10'))].task.nunique()}, "
      f"libero_90 {E[E.task.str.startswith('libero_90')].task.nunique()}) | pairs kept: {len(P)} ({int(P.significant_18pp.sum())} significant)")
print(E.category.value_counts().to_string())
