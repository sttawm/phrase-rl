#!/usr/bin/env python3
"""Reproduce every number the v12 spec rests on, from committed data.

Written to be audited: each block prints the claim, the computation, and the
result, so a reader can check the reasoning rather than trust the summary.
Several claims in the first draft of ALGORITHM-RL-V12.md were WRONG and were
overturned by exactly these computations -- the ones marked [CORRECTION].

    .venv/bin/python scripts/audit_v11_evidence.py
    .venv/bin/python scripts/audit_v11_evidence.py --replay <path/to/replay.pt>

The --replay blocks need a v11 checkpoint's replay.pt (candidate TEXT and
per-candidate rewards, 50-step window). Recover one with:
    bash scripts/ckpt_archive.sh unpack v11_step_0150
"""
import argparse
import glob
import json
import re

import numpy as np
import pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--replay", default="", help="path to a v11 replay.pt")
args = ap.parse_args()

PROXY = {"C": 8.123, "bz": 0.4445, "bg": 11.3193}
F_CAP = 4


def rank01(x):
    """Average-rank in [0,1] -- the transform used by blend_rewards (c4b)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 1:
        return np.full(n, 0.5)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(n)
    r[o] = np.arange(n, dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r / (n - 1)


def zsc(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def advantage(r):
    """Exactly as apply_update computes it."""
    r = np.asarray(r, dtype=float)
    return (r - r.mean()) / (r.std() + 1e-6)


def hdr(n, claim):
    print(f"\n{'=' * 78}\n[{n}] {claim}\n{'=' * 78}")


# --------------------------------------------------------------- ground truth
feats = pd.concat([pd.read_parquet("results/analysis/fine_exam_features_native.parquet"),
                   pd.read_parquet("results/analysis/fine_exam_features_oov.parquet")],
                  ignore_index=True)
pan = pd.read_parquet("results/analysis/fine_exam_phrases.parquet")
rows = []
for (t, p), g in feats.groupby(["task", "phrase"]):
    pe = (g.sort_values("t").groupby("episode_index").head(F_CAP)
           .groupby("episode_index").agg(z=("z_row", "mean"), gr=("grip_row", "mean")))
    rows.append({"task": t, "phrase": p, "z": pe.z.mean(), "negg": -pe.gr.mean()})
X = pd.DataFrame(rows).merge(pan[["task", "phrase", "gt_success"]], on=["task", "phrase"])


hdr(1, "v11 rollout trend: is it flat, and what could we have detected?")
cells = []
for f in sorted(glob.glob("results/analysis/v11cells/nat24_0*.json")):
    d = json.load(open(f))
    cells.append((int(re.search(r"nat24_(\d+)", f).group(1)), d["pooled"], d["n"]))
cells.sort()
s = np.array([c[0] for c in cells], dtype=float)
y = np.array([c[1] for c in cells])
n_ep = cells[0][2]
res = stats.linregress(s, y)
se_cell = np.sqrt((y.mean() / 100) * (1 - y.mean() / 100) / n_ep) * 100
print(f"  {len(cells)} cells, {n_ep} episodes each, steps {int(s.min())}-{int(s.max())}")
print(f"  slope {res.slope:+.4f} pp/step  95% CI [{res.slope-1.96*res.stderr:+.4f}, "
      f"{res.slope+1.96*res.stderr:+.4f}]  p={res.pvalue:.3f}")
print(f"  per-cell 1 s.e. = {se_cell:.2f}pp")
print(f"  MINIMUM DETECTABLE slope = {1.96*res.stderr:.4f} pp/step "
      f"= {1.96*res.stderr*1000:.1f}pp per 1000 steps")
print("  -> improvement slower than that is INVISIBLE at n=192. This is why the")
print("     v12 spec calls for 768-episode cells.")


hdr(2, "Transfer: the reward improved -- how much SHOULD rollouts have moved?")
dz, dgrip = 0.545, 0.0046          # nominal tier, first20 vs last20 of train_log
dlog = PROXY["bz"] * dz + PROXY["bg"] * dgrip
base = y[0] / 100
lo = np.log(base / (1 - base))
print(f"  verifier gain dz={dz:+.3f} -> logit {PROXY['bz']*dz:+.3f}")
print(f"  gripper  gain dg={dgrip:+.4f} -> logit {PROXY['bg']*dgrip:+.3f}")
print(f"  predicted rollout change from the {y[0]:.2f}% baseline: "
      f"{1/(1+np.exp(-(lo+dlog)))*100 - y[0]:+.1f}pp")
print(f"  observed over the measured range: {res.slope*s.max():+.1f}pp "
      f"(s.e. {res.stderr*s.max():.1f}pp)")
print("  CAVEAT, and it is the finding: these coefficients were fitted ACROSS")
print("  PHRASES, not on policy-induced change. Their failure to predict is the")
print("  evidence for off-manifold optimisation, not a refutation of the proxy.")


hdr(3, "Reward form: within-task fidelity vs REAL rollout success")
forms = {
    "c4b rank01 w=0.25 (v11)": lambda g: 0.25 * rank01(g.z) + 0.75 * rank01(g.negg),
    "raw-std w=0.25": lambda g: 0.25 * zsc(g.z) + 0.75 * zsc(g.negg),
    "raw-std w=0.75": lambda g: 0.75 * zsc(g.z) + 0.25 * zsc(g.negg),
    "proxy logit": lambda g: PROXY["bz"] * g.z.values + PROXY["bg"] * g.negg.values,
    "gripper only": lambda g: g.negg.values,
    "verifier only": lambda g: g.z.values,
}
print(f"  {len(X)} phrases, {X.task.nunique()} tasks, 36 real rollouts each")
print(f"  {'form':26s} {'Pearson':>8s} {'Spearman':>9s} {'worst task':>11s}")
for k, fn in forms.items():
    pe, sp = [], []
    for t, g in X.groupby("task"):
        if len(g) < 5:
            continue
        f = fn(g)
        pe.append(np.corrcoef(f, g.gt_success)[0, 1])
        sp.append(stats.spearmanr(f, g.gt_success)[0])
    print(f"  {k:26s} {np.nanmean(pe):+8.3f} {np.nanmean(sp):+9.3f} {np.nanmin(pe):+11.3f}")
print("  [CORRECTION] w=0.75 is WORSE than w=0.25 within group; the fine_grid case")
print("  for 0.75 is a cross-task large-gap effect the update never sees.")


hdr(4, "Degenerate group: which reward form refuses to invent gradient?")
rng = np.random.default_rng(0)
K = 16
z = np.full(K, 0.5) + rng.normal(0, 1e-7, K)
ng = np.full(K, -0.05) + rng.normal(0, 1e-7, K)
print("  16 candidates identical to within 1e-7 (batched GPU reductions are not")
print("  bit-deterministic, so identical text does NOT give identical channels)")
for k, r in [("c4b rank01", 0.25 * rank01(z) + 0.75 * rank01(ng)),
             ("raw-std", 0.25 * zsc(z) + 0.75 * zsc(ng)),
             ("proxy logit", PROXY["bz"] * z + PROXY["bg"] * ng)]:
    print(f"  {k:14s} reward spread {r.max()-r.min():9.2e}   "
          f"max |advantage| {np.abs(advantage(r)).max():.3f}")
print("  -> only fixed coefficients keep eps at eps until GRPO's 1e-6 floor caps it.")
print("  [CORRECTION] raw-std does NOT fix this; it is marginally worse than rank01,")
print("  because dividing by the group's own std re-inflates eps to unit variance.")


hdr(5, "Best-of-K: does the current reward get better or worse with more candidates?")
rng = np.random.default_rng(0)
for k in (2, 4, 8, 16):
    prox, orac = [], []
    for t, g in X.groupby("task"):
        if len(g) < k:
            continue
        zz, gg, yy = g.z.values, g.negg.values, g.gt_success.values
        for _ in range(400):
            i = rng.choice(len(g), size=k, replace=False)
            f = 0.25 * rank01(zz[i]) + 0.75 * rank01(gg[i])
            prox.append(yy[i][int(np.argmax(f))] - yy[i].mean())
            orac.append(yy[i].max() - yy[i].mean())
    print(f"  K={k:2d}  proxy-best {np.mean(prox):+5.2f}pp   oracle-best {np.mean(orac):+5.2f}pp")
print("  -> proxy-best PEAKS at K=4 and declines; the oracle keeps rising. Adding")
print("     candidates widens the gap between available and findable.")
print("  [CORRECTION] an earlier claim that K=16 is WORSE THAN RANDOM does not")
print("  replicate on the full set (it used the native subset only).")


hdr(6, "Context budget: is C=5 -> C=10 worth it?")
grid = json.load(open("results/analysis/fine_grid_all.json"))["grid"]
ks = sorted([k for k in grid if k.startswith("F4_")],
            key=lambda k: int(re.search(r"C(\d+)", k).group(1)))
print("  F=4, c4b.      C   5-10pp pairs   15+pp pairs")
for k in ks:
    c = grid[k].get("c4b", {})
    cnum = int(re.search(r"C(\d+)", k).group(1))
    fine = c.get("fine_5-10", float("nan"))
    far = c.get("far_15+", float("nan"))
    print("  %14d %14.1f %13.1f" % (cnum, fine, far))
print("  -> close pairs are what a GRPO group contains. C=5->10 buys ~+5pp there;")
print("     saturation begins only AFTER C=10.")
print("  [CORRECTION] the claim that C=5->10 is saturated read the far_15+ column.")


if args.replay:
    import torch
    from collections import defaultdict

    hdr(7, "Candidate diversity and the reward noise floor (needs replay.pt)")
    G = torch.load(args.replay, map_location="cpu", weights_only=False)["groups"]

    def norm(s):
        s = s.lower().strip().rstrip(".!?")
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s))

    raw = [len(set(g["survivors"])) for g in G]
    soft = [len(set(norm(s) for s in g["survivors"])) for g in G]
    wset = [len(set(frozenset(norm(s).split()) for s in g["survivors"])) for g in G]
    print(f"  {len(G)} groups of 16")
    print(f"  unique raw strings     : mean {np.mean(raw):.1f} median {int(np.median(raw))}")
    print(f"  after case/punct norm  : mean {np.mean(soft):.1f} median {int(np.median(soft))}")
    print(f"  as unordered word sets : mean {np.mean(wset):.1f} median {int(np.median(wset))}")

    noise, exact_spreads = [], []
    for g in G:
        by, byx = defaultdict(list), defaultdict(list)
        for s, r in zip(g["survivors"], g["rewards"]):
            by[norm(s)].append(float(r))
            byx[s].append(float(r))
        noise += [max(v) - min(v) for v in by.values() if len(v) > 1]
        exact_spreads += [max(v) - min(v) for v in byx.values() if len(v) > 1]
    tot = [max(map(float, g["rewards"])) - min(map(float, g["rewards"])) for g in G]
    noise = np.array(noise)
    ex = np.array(exact_spreads)
    print(f"\n  reward spread on text identical after normalisation (NOISE FLOOR):")
    print(f"    n={len(noise)} mean {noise.mean():.3f} median {np.median(noise):.3f} "
          f"p90 {np.percentile(noise,90):.3f}")
    print(f"  total within-group spread: mean {np.mean(tot):.3f}")
    print(f"  -> noise floor is {100*noise.mean()/np.mean(tot):.0f}% of what the update treats as signal")
    print(f"\n  EXACT duplicates with a NONZERO spread: "
          f"{int((ex>0).sum())}/{len(ex)} ({100*(ex>0).mean():.1f}%)")
    print("  every nonzero value is a multiple of a rank step (0.25/16, 0.75/16, 1/16)")
    print("  -> tie-breaking on float noise, not differing frames or seeds.")
else:
    print("\n(blocks needing replay.pt skipped; pass --replay <path>)")

print("\nDone. Numbers here supersede any summary that disagrees with them.")
