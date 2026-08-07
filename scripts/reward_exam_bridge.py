#!/usr/bin/env python3
"""Unified recomputation of the 92-vs-68 gripper 'coarse' discrepancy.

One estimator (CRN bootstrap: C episodes shared per task, F frames per episode,
grip channel, rank01 within task, pair sign accuracy at F=4 C=20), applied to a
ladder of pair populations that walks from the native exam's 92% to the
stratified exam's 68%. Every step changes exactly one ingredient:

  A  native gate-zero pairs (fc_grid bank, gaps>=6pp z>1.96)      [the 92]
  B  ... restricted to gap >= 15pp (stratified exam's definition)
  C  fine-exam bank, far_15+ pairs, NATIVE phrases only
  D  fine-exam bank, far_15+ pairs, ALL (incl. OOV)               [the 68]
  E  D without widowx_coke_can_on_plate_clean (inversion task)
  F  D with both-plausible filter (min success >= 10%)

  python3 scripts/reward_exam_bridge.py
"""
import itertools
import json
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "results/analysis")
import reward_bakeoff_v3 as bk  # noqa: E402

B = 400
F, C = 4, 20
RNG = np.random.default_rng(7)


def crn_grip_acc(states, pairs, rng):
    """states: task -> dict(G[key,ep,t], ki, n_eps, t_avail). pairs: (task, better_key, worse_key)."""
    hits = tot = 0
    for _ in range(B):
        per_task = {}
        for t, s in states.items():
            elig = [e for e in range(s["n_eps"]) if len(s["t_avail"][e]) >= 1]
            epick = rng.choice(elig, size=min(C, len(elig)), replace=False)
            gsel = []
            for e in epick:
                avail = s["t_avail"][e]
                fp = rng.choice(avail, size=min(F, len(avail)), replace=False)
                gsel.append(np.nanmean(s["G"][:, e, :][:, fp], axis=1))
            g = np.nanmean(np.stack(gsel), axis=0)
            order = np.argsort(np.argsort(-g))
            per_task[t] = order / max(len(g) - 1, 1)
    # (loop below re-uses the LAST draw only if we scored inside; score inside instead)
        for t, bkey, wkey in pairs:
            if t not in per_task:
                continue
            sc = per_task[t]
            sb, sw = sc[states[t]["ki"][bkey]], sc[states[t]["ki"][wkey]]
            if not (np.isnan(sb) or np.isnan(sw)):
                hits += int(sb > sw)
                tot += 1
    return 100 * hits / tot, tot // B


def build_states(feats, keycol):
    states = {}
    for t, sub in feats.groupby("task"):
        keys = sorted(sub[keycol].unique())
        eps = sorted(sub.episode_index.unique())
        ts = sorted(sub.t.unique())
        G = np.full((len(keys), len(eps), len(ts)), np.nan)
        ki = {k: i for i, k in enumerate(keys)}
        ei = {e: i for i, e in enumerate(eps)}
        ti = {x: i for i, x in enumerate(ts)}
        for r in sub.itertuples():
            G[ki[getattr(r, keycol)], ei[r.episode_index], ti[r.t]] = r.grip_row
        t_avail = {ei[e]: np.where(~np.isnan(G[:, ei[e], :]).all(axis=0))[0] for e in eps}
        states[t] = dict(G=G, ki=ki, n_eps=len(eps), t_avail=t_avail)
    return states


results = {}

# ---- legs A/B: native exam bank -------------------------------------------
feats = bk.load_features()
feats["grip_row"] = [float(np.mean(v)) for v in feats.grip_err]
feats["key"] = [bk.norm_key(p) for p in feats.phrase]
nat_states = build_states(feats, "key")

gz = json.load(open("results/analysis/gate_zero_pairs.json"))["pairs"]
cov = [p for p in gz if p["task"] in nat_states
       and bk.norm_key(p["better"]) in nat_states[p["task"]]["ki"]
       and bk.norm_key(p["worse"]) in nat_states[p["task"]]["ki"]]
pairs_a = [(p["task"], bk.norm_key(p["better"]), bk.norm_key(p["worse"])) for p in cov]
gaps = [100 * abs(p["better_succ"] - p["worse_succ"]) for p in cov]
acc, n = crn_grip_acc(nat_states, pairs_a, np.random.default_rng(7))
results["A_native_gate_zero"] = dict(acc=round(acc, 1), n=n, median_gap=round(float(np.median(gaps)), 1))

pairs_b = [pa for pa, g in zip(pairs_a, gaps) if g >= 15]
acc, n = crn_grip_acc(nat_states, pairs_b, np.random.default_rng(7))
results["B_native_gap15plus"] = dict(acc=round(acc, 1), n=n)

# ---- legs C-F: fine-exam bank ---------------------------------------------
fnat = pd.read_parquet("results/analysis/fine_exam_features_native.parquet")
foov = pd.read_parquet("results/analysis/fine_exam_features_oov.parquet")
pan = pd.read_parquet("results/analysis/fine_exam_phrases.parquet")
gt = {(r.task, r.phrase): (r.gt_success, r.gt_n) for r in pan.itertuples()}


def far_pairs(states, min_succ=None, drop_task=None):
    out = []
    for t, s in states.items():
        if drop_task and t == drop_task:
            continue
        ph = [(p, *gt[(t, p)]) for p in s["ki"] if (t, p) in gt]
        for (p1, s1, n1), (p2, s2, n2) in itertools.combinations(ph, 2):
            gap = abs(s1 - s2)
            se = 100 * math.sqrt(s1 / 100 * (1 - s1 / 100) / n1 + s2 / 100 * (1 - s2 / 100) / n2)
            conf = 0.5 * (1 + math.erf((gap / se) / math.sqrt(2))) if se > 0 else 1.0
            if gap >= 15 and conf >= 0.8:
                if min_succ is not None and min(s1, s2) < min_succ:
                    continue
                out.append((t, p1 if s1 > s2 else p2, p2 if s1 > s2 else p1))
    return out

nat_fine_states = build_states(fnat, "phrase")
acc, n = crn_grip_acc(nat_fine_states, far_pairs(nat_fine_states), np.random.default_rng(11))
results["C_finebank_far15_native_only"] = dict(acc=round(acc, 1), n=n)

all_states = build_states(pd.concat([fnat, foov], ignore_index=True), "phrase")
acc, n = crn_grip_acc(all_states, far_pairs(all_states), np.random.default_rng(11))
results["D_finebank_far15_all"] = dict(acc=round(acc, 1), n=n)

acc, n = crn_grip_acc(all_states, far_pairs(all_states, drop_task="widowx_coke_can_on_plate_clean"),
                      np.random.default_rng(11))
results["E_D_minus_cokeplate"] = dict(acc=round(acc, 1), n=n)

acc, n = crn_grip_acc(all_states, far_pairs(all_states, min_succ=10), np.random.default_rng(11))
results["F_D_bothplausible"] = dict(acc=round(acc, 1), n=n)

print(f"\ngrip-only, F={F} C={C}, B={B}, CRN estimator identical across legs:")
for k, v in results.items():
    extra = f"  median_gap={v['median_gap']}" if "median_gap" in v else ""
    print(f"  {k:34s} acc={v['acc']:5.1f}  pairs={v['n']}{extra}")
json.dump(results, open("results/analysis/reward_exam_bridge.json", "w"), indent=1)
print("-> results/analysis/reward_exam_bridge.json")
