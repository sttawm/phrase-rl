#!/usr/bin/env python3
"""LIBERO sealed v2, round 1: no-rephraser baseline vs every apply arm.

  .venv/bin/python scripts/analyze_libero_v2.py [--run p_v2] [--prefix v2r1]
      [--applies-dir results/analysis/pi05_bank/eval_applies_v2]

Loads every results/rules_runs/<run>/jobs/<prefix>*.result.parquet into a
(task, phrase) -> rolled success table (gt_success, 0-100 over the inits;
duplicate rows across legs are pooled by n_ctx). Bases come from
results/analysis/pi05_bank/naturals_v2.parquet (task, phrase); each parquet in
--applies-dir is one arm (task, phrase=base, rewrite, applier, book). An arm's
success for a base is the rolled success of its rewrite string (an empty
rewrite means the base rolled unchanged); the no-rephraser arm is the base
itself. Reports per arm the base-weighted mean success pooled and split
in-finetune (suite != libero_90) vs out-of-finetune, the fraction of bases the
arm left unchanged, the number of bases without a rollout, and a
paired-by-base sign-flip permutation p of each arm vs the no-rephraser
baseline and vs the scaffold arm (book name containing "scaffold"). Writes
results/analysis/pi05_bank/libero_v2_round1_cells.json. With no result files
present it prints "no results yet" and exits 0.
"""
import argparse
import glob
import itertools
import json
import pathlib

import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
B = R / "results/analysis/pi05_bank"
OUT = B / "libero_v2_round1_cells.json"
N_PERM = 10000
rng = np.random.default_rng(20260914)


def norm_task(t):
    """Analysis files key tasks 'suite/id', the job queue 'suite:id'; use ':'."""
    return str(t).strip().replace("/", ":")


def suite_of(task):
    return task.split(":")[0]


def load_results(run, prefix):
    files = sorted(glob.glob(str(R / f"results/rules_runs/{run}/jobs/{prefix}*.result.parquet")))
    if not files:
        return files, None
    res = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    res = res.dropna(subset=["gt_success"])
    res["task"] = res.task.map(norm_task)
    res["phrase"] = res.phrase.astype(str).str.strip()
    res["n_ctx"] = res.n_ctx.fillna(1).astype(float)
    res["w"] = res.gt_success * res.n_ctx
    g = res.groupby(["task", "phrase"]).agg(w=("w", "sum"), n=("n_ctx", "sum"))
    return files, (g.w / g.n).rename("succ")


def load_arms(applies_dir, bases):
    """(label, table indexed by (task, base) with column 'phrase' = rolled string)."""
    arms = []
    for f in sorted(glob.glob(str(applies_dir / "*.parquet"))):
        d = pd.read_parquet(f)
        d["task"] = d.task.map(norm_task)
        d = d.rename(columns={"phrase": "base", "rewrite": "phrase"})
        # keys are stripped to match the stager contract (load_pairs strips before rolling)
        d["base"] = d.base.astype(str).str.strip()
        d["phrase"] = d.phrase.fillna("").astype(str).str.strip()
        empty = d.phrase == ""
        d.loc[empty, "phrase"] = d.base[empty]
        d = d.drop_duplicates(["task", "base"])
        label = f"{d.book.iloc[0]}|{d.applier.iloc[0]}" if len(d) else pathlib.Path(f).stem
        d = d.merge(bases.rename(columns={"phrase": "base"}), on=["task", "base"])
        arms.append((label, d.set_index(["task", "base"])[["phrase"]]))
    return arms


def sign_flip_p(d):
    d = np.asarray(d, float)
    if len(d) == 0:
        return float("nan")
    if len(d) <= 12:
        signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
        null = np.abs((signs * d).mean(axis=1))
        return float((null >= abs(d.mean()) - 1e-12).mean())
    signs = rng.choice([1, -1], size=(N_PERM, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float(((null >= abs(d.mean()) - 1e-12).sum() + 1) / (N_PERM + 1))


def splits(idx):
    out_mask = np.array([suite_of(t) == "libero_90" for t, _ in idx], dtype=bool)
    return {"pooled": np.ones(len(idx), bool), "in": ~out_mask, "out": out_mask}


def paired(arm_succ, ref_succ):
    """delta/p per split over bases rated in both arms."""
    j = pd.concat([arm_succ.rename("a"), ref_succ.rename("r")], axis=1).dropna()
    if j.empty:  # no base rated in both arms yet (e.g. rewrite legs still pending)
        return {f"{k}_{s}": (0 if k == "n" else None)
                for s in ("pooled", "in", "out") for k in ("delta", "p", "n")}
    rec = {}
    for name, m in splits(j.index).items():
        d = (j.a - j.r).values[m]
        rec[f"delta_{name}"] = round(float(d.mean()), 2) if len(d) else None
        rec[f"p_{name}"] = round(sign_flip_p(d), 4) if len(d) else None
        rec[f"n_{name}"] = int(len(d))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p_v2")
    ap.add_argument("--prefix", default="v2r1")
    ap.add_argument("--applies-dir", default=str(B / "eval_applies_v2"))
    ap.add_argument("--bases", default=str(B / "naturals_v2.parquet"))
    a = ap.parse_args()

    files, rates = load_results(a.run, a.prefix)
    if rates is None:
        print(f"no results yet (results/rules_runs/{a.run}/jobs/{a.prefix}*.result.parquet)")
        return 0
    print(f"result files: {len(files)}; rated (task, phrase): {len(rates)}")

    bases = pd.read_parquet(a.bases)[["task", "phrase"]]
    bases["task"] = bases.task.map(norm_task)
    bases["phrase"] = bases.phrase.astype(str).str.strip()
    bases = bases.drop_duplicates(["task", "phrase"]).reset_index(drop=True)
    print(f"bases: {len(bases)} over {bases.task.nunique()} tasks")

    # every arm, the no-rephraser baseline first (its rolled string is the base)
    none = bases.rename(columns={"phrase": "base"}).assign(phrase=lambda d: d.base)
    arms = [("none", none.set_index(["task", "base"])[["phrase"]])]
    arms += load_arms(pathlib.Path(a.applies_dir), bases)
    scaffold = [lab for lab, _ in arms if "scaffold" in lab.split("|")[0].lower()]
    if len(scaffold) > 1:
        print(f"WARNING: several scaffold arms {scaffold}; using {scaffold[0]}")
    scaffold = scaffold[0] if scaffold else None

    succ = {}
    cells = {}
    for label, m in arms:
        s = m.join(rates, on=["task", "phrase"]).succ
        succ[label] = s
        rec = {"n_base": int(len(m)), "n_missing": int(s.isna().sum()),
               "unchanged_frac": round(float(
                   (m.phrase == m.index.get_level_values("base")).mean()), 3)}
        for name, mask in splits(m.index).items():
            sub = s[mask].dropna()
            rec[f"mean_{name}"] = round(float(sub.mean()), 2) if len(sub) else None
            rec[f"n_{name}"] = int(len(sub))
        cells[label] = rec

    for label in cells:
        if label != "none":
            cells[label]["vs_none"] = paired(succ[label], succ["none"])
        if scaffold and label not in ("none", scaffold):
            cells[label]["vs_scaffold"] = paired(succ[label], succ[scaffold])

    OUT.write_text(json.dumps({"run": a.run, "prefix": a.prefix, "scaffold_arm": scaffold,
                               "arms": cells}, indent=1))
    print("cells ->", OUT)

    def fmt(v, w=6, prec=1):
        return f"{v:{w}.{prec}f}" if isinstance(v, (int, float)) else " " * (w - 1) + "-"

    print(f"\n{'arm':28s} {'pooled':>6} {'in':>6} {'out':>6}  {'nB':>4} {'unch':>5} {'miss':>4}  "
          f"{'Δnone':>6} {'p':>7}  {'Δscaf':>6} {'p':>7}")
    for label, c in cells.items():
        vn, vs = c.get("vs_none", {}), c.get("vs_scaffold", {})
        print(f"{label:28s} {fmt(c['mean_pooled'])} {fmt(c['mean_in'])} {fmt(c['mean_out'])}  "
              f"{c['n_base']:4d} {c['unchanged_frac']:5.2f} {c['n_missing']:4d}  "
              f"{fmt(vn.get('delta_pooled'))} {fmt(vn.get('p_pooled'), 7, 4)}  "
              f"{fmt(vs.get('delta_pooled'))} {fmt(vs.get('p_pooled'), 7, 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
