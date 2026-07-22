"""POST-HOC EXPLORATORY addendum to the FROZEN reward_bakeoff_v3 (2026-07-22).

Question (user): did any examined reward combine gripper error with full-action-
chunk L2? Answer: no — norm_l2 was only a learned-verifier input feature. This
addendum grades analytic full-chunk L2 (row-mean norm_l2 vs a*, all 7 dims)
alone and in rank-blends with grip, on the SAME frozen features, pairs, success
table, and exam definitions as the bakeoff. It changes no decision: C4b remains
the pre-registered winner; this measures whether arm dims add signal to grip.

Candidates (all rank01-space, higher=better):
  C1_ens4f        incumbent learned ensemble (recomputed control)
  C4b_rank_25_75  frozen winner 0.25*z01+0.75*g01 (recomputed control)
  GRIP_pure       rank01(-grip)              — the exam ceiling control
  L2_pure         rank01(-l2)                — the analytic "L2 arm" reward
  GL_75_25/50_50/25_75  w*g01+(1-w)*l01      — grip x full-L2 blends
  ZL_25_75        0.25*z01+0.75*l01          — C4b with L2 in grip's seat

Run:  .venv/bin/python results/analysis/reward_exam_l2_addendum.py
Out:  results/analysis/reward_exam_l2_addendum.json
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "results/analysis")
import reward_bakeoff_v3 as bk  # noqa: E402  (import chdirs to repo root)

B = 2000
RNG = np.random.default_rng(0)

feats = bk.load_features()
ens = bk.VerifierEnsemble("results/checkpoints/verifier_reward_ensemble_4f.json")
feats["z_row"] = ens.member_logits(feats).mean(axis=1)
feats["grip_row"] = [float(np.mean(v)) for v in feats.grip_err]
feats["l2_row"] = [float(np.mean(v)) for v in feats.norm_l2]


class S:
    pass


states = {}
for t, sub in feats.groupby("task"):
    sub = sub.assign(key=[bk.norm_key(p) for p in sub.phrase])
    s = S()
    s.keys = sorted(sub.key.unique())
    s.eps = sorted(sub.episode_index.unique())

    def piv(col, sub=sub, s=s):
        p = sub.pivot_table(index="key", columns="episode_index", values=col, aggfunc="mean")
        return p.loc[s.keys, s.eps].values

    s.Z, s.G, s.L = piv("z_row"), piv("grip_row"), piv("l2_row")
    assert not (np.isnan(s.Z).any() or np.isnan(s.G).any() or np.isnan(s.L).any()), t
    s.kidx = {k: i for i, k in enumerate(s.keys)}
    states[t] = s
assert set(states) == set(bk.NATIVE)

r01 = bk.rank01


def cands(z, g, l):
    z01, g01, l01 = r01(z), r01(-g), r01(-l)
    return {
        "C1_ens4f": z,
        "C4b_rank_25_75": 0.25 * z01 + 0.75 * g01,
        "GRIP_pure": g01,
        "L2_pure": l01,
        "GL_75_25": 0.75 * g01 + 0.25 * l01,
        "GL_50_50": 0.5 * g01 + 0.5 * l01,
        "GL_25_75": 0.25 * g01 + 0.75 * l01,
        "ZL_25_75": 0.25 * z01 + 0.75 * l01,
    }


CANDS = ["C1_ens4f", "C4b_rank_25_75", "GRIP_pure", "L2_pure",
         "GL_75_25", "GL_50_50", "GL_25_75", "ZL_25_75"]

point = {t: cands(s.Z.mean(axis=1), s.G.mean(axis=1), s.L.mean(axis=1))
         for t, s in states.items()}

# ---- gate-zero signs on the same covered frozen pairs ----
pairs = json.load(open("results/analysis/gate_zero_pairs.json"))["pairs"]
covered = [p for p in pairs if p["task"] in states
           and bk.norm_key(p["better"]) in states[p["task"]].kidx
           and bk.norm_key(p["worse"]) in states[p["task"]].kidx]
signs = {c: {"correct": 0, "n": 0, "fails": []} for c in CANDS}
for p in covered:
    s = states[p["task"]]
    for c in CANDS:
        sb = float(point[p["task"]][c][s.kidx[bk.norm_key(p["better"])]])
        sw = float(point[p["task"]][c][s.kidx[bk.norm_key(p["worse"])]])
        ok = sb > sw
        signs[c]["n"] += 1
        signs[c]["correct"] += int(ok)
        if not ok:
            signs[c]["fails"].append({"task": p["task"], "better": p["better"],
                                      "worse": p["worse"], "delta_pp": p["delta_pp"],
                                      "margin": round(sb - sw, 4)})

# ---- top-1/3 regret (point + scorefix bootstrap) + spearman, per v3 ----
succ_tab = pd.read_parquet("results/analysis/phrase_success_table.parquet")
succ_tab["key"] = [bk.norm_key(p) for p in succ_tab.phrase]


def regret_from(sv, kidxs, succ, top=1):
    order = np.argsort(-sv[kidxs])
    return float(succ.max() - succ[order[:top]].max())


regret, spear = {}, {}
pooled = {c: {"a": [], "b": []} for c in CANDS}
for t in bk.NATIVE:
    s = states[t]
    sub = succ_tab[(succ_tab.task == t) & succ_tab.key.isin(s.kidx)].drop_duplicates("key")
    keys = list(sub.key)
    succ = sub.set_index("key").loc[keys].succ.values * 100
    kidxs = np.array([s.kidx[k] for k in keys])
    boot = {c: [] for c in CANDS}
    n_e = len(s.eps)
    for _ in range(B):
        eidx = RNG.choice(n_e, n_e)
        sc = cands(s.Z[:, eidx].mean(axis=1), s.G[:, eidx].mean(axis=1),
                   s.L[:, eidx].mean(axis=1))
        for c in CANDS:
            boot[c].append(regret_from(sc[c], kidxs, succ, 1))
    regret[t] = {}
    for c in CANDS:
        regret[t][c] = {
            "top1_point": round(regret_from(point[t][c], kidxs, succ, 1), 1),
            "top3_point": round(regret_from(point[t][c], kidxs, succ, 3), 1),
            "top1_scorefix_p95": round(float(np.percentile(boot[c], 95)), 1),
        }
        sv = point[t][c][kidxs]
        pooled[c]["a"] += list(r01(sv))
        pooled[c]["b"] += list(r01(succ))
        spear.setdefault(c, {})[t] = round(bk.spearman(sv, succ), 3)

rows = []
for c in CANDS:
    g = signs[c]
    per = {t: regret[t][c]["top1_point"] for t in bk.NATIVE}
    p95 = {t: regret[t][c]["top1_scorefix_p95"] for t in bk.NATIVE}
    rows.append({
        "candidate": c,
        "signs": f"{g['correct']}/{g['n']}",
        "max_top1_regret": max(per.values()),
        "max_top1_scorefix_p95": max(p95.values()),
        "top1_per_task": per,
        "spearman_pooled": round(bk.spearman(pooled[c]["a"], pooled[c]["b"]), 3),
    })

print(f"\nL2 ADDENDUM — {len(covered)} covered frozen pairs, B={B} scorefix boots")
print(f"{'candidate':16s} {'signs':>7s} {'maxTop1':>8s} {'maxP95':>7s} {'spear':>6s}")
for r in rows:
    print(f"{r['candidate']:16s} {r['signs']:>7s} {r['max_top1_regret']:8.1f} "
          f"{r['max_top1_scorefix_p95']:7.1f} {r['spearman_pooled']:6.3f}")
print("\nfails per candidate:")
for c in CANDS:
    for f in signs[c]["fails"]:
        print(f"  {c}: [{f['task'].replace('widowx_', '')}] margin={f['margin']:+.3f} "
              f"delta={f['delta_pp']}pp  '{f['better'][:40]}' > '{f['worse'][:40]}'")

json.dump({"note": "POST-HOC EXPLORATORY; frozen bakeoff decisions unchanged",
           "covered_pairs": len(covered), "B": B, "verdict_rows": rows,
           "signs": {c: {k: v for k, v in signs[c].items()} for c in CANDS},
           "regret_per_task": regret, "spearman_per_task": spear},
          open("results/analysis/reward_exam_l2_addendum.json", "w"), indent=1)
print("\nwrote results/analysis/reward_exam_l2_addendum.json")
