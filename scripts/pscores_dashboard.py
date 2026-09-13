#!/usr/bin/env python3
"""Paired-by-base p-values for the dashboard's row-1 bars vs no-rephraser.

Each dashboard bar (condition x applier x diet, mean over draws; plus the
scaffold bars) is a base-weighted mean of per-base success rates. The matching
test: per base, (cell rate - baseline rate) over the same 24-layout grid, then
a sign-flip permutation test (exact when bases <= 12, else 20k resamples) on
the mean of those paired differences, with a paired t as a cross-check. This
respects the base clustering (the pooling unit) and the CRN layout pinning
that makes cell and baseline comparable per base.

  LEGS_DIR=<dir with *.result.parquet> AP_DIR=<dir with ph_*.parquet> \
    .venv/bin/python scripts/pscores_dashboard.py

Legs fetched from origin (results/rules_runs/r1_sim/jobs/) by the session's
fetch_legs.py; baselines: d34basen legs (natural), ever72_off jsonls
(adversarial, A37 passthrough re-roll, 24x1), sens_dualmetric orig_pass
(original, 24x1). Output: results/analysis/dashboard_pscores.json + .png.
"""
import glob
import itertools
import json
import math
import os
import pathlib

import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
LEGS = os.environ["LEGS_DIR"]
AP = os.environ["AP_DIR"]
rng = np.random.default_rng(20260911)


def leg_pool(pattern):
    fs = glob.glob(f"{LEGS}/{pattern}")
    assert fs, pattern
    L = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return L.groupby(["task", "phrase"], as_index=False).agg(
        succ=("gt_success", "mean"))


def per_base(leg_pat, apply_pq):
    """base -> success rate: leg rates joined through the apply table.
    Applies without a rolled leg row (the 6 judge-dropped naturals in the
    192-row a34/a36 tables) are dropped; anything beyond that asserts."""
    L = leg_pool(leg_pat)
    a = pd.read_parquet(f"{AP}/{apply_pq}")[["task", "base", "phrase"]]
    m = a.merge(L, on=["task", "phrase"], how="left")
    n_miss = int(m.succ.isna().sum())
    assert n_miss <= 6, f"{apply_pq}: {n_miss} unmatched applies"
    return m[m.succ.notna()].groupby(["task", "base"]).succ.mean()


def sign_flip_p(d):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    n = len(d)
    if n <= 12:                      # exact: all 2^n sign patterns
        signs = np.array(list(itertools.product([1, -1], repeat=n)))
        null = np.abs((signs * d).mean(axis=1))
        return float((null >= obs - 1e-12).mean())
    signs = rng.choice([1, -1], size=(20000, n))
    null = np.abs((signs * d).mean(axis=1))
    return float(((null >= obs - 1e-12).sum() + 1) / (len(null) + 1))


def paired_t_p(d):
    d = np.asarray(d, float)
    se = d.std(ddof=1) / math.sqrt(len(d))
    if se == 0:
        return 1.0
    t = d.mean() / se
    # two-sided p via survival of t-dist; normal approx is fine at n>=12
    from math import erfc, sqrt
    return erfc(abs(t) / sqrt(2)) if len(d) > 60 else _t_sf(abs(t), len(d) - 1)


def _t_sf(t, df):
    # two-sided student-t p via incomplete beta (no scipy in .venv)
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    # continued-fraction regularized incomplete beta I_x(a, b)
    def betacf(a, b, x):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        h = d
        for m in range(1, 200):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
            de = d * c
            h *= de
            if abs(de - 1.0) < 3e-9:
                break
        return h
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b + lbeta) / a
    ib = front * betacf(a, b, x) if x < (a + 1) / (a + b + 2) else \
        1 - math.exp(math.log(1 - x) * b + math.log(x) * a + lbeta) / b * betacf(b, a, 1 - x)
    return ib


# ---- baselines: base -> rate ------------------------------------------------
base_rates = {}
Lb = leg_pool("d34basen_*.result.parquet")
base_rates["nat"] = Lb.set_index(["task", "phrase"]).succ
base_rates["nat"].index.names = ["task", "base"]

# adversarial + original baselines: the PUBLISHED baseline rolls (the same
# data behind the 24.5 / 36.1 dashboard constants): a29pass legs (60 new
# attacks x 24 x 1) + anchors_x12.parquet passthrough (original 12 attacks
# x 24 x 12) and originals (12 canonicals x 24 x 12). anchors_x12 fetched
# to LEGS_DIR's parent by the session fetch step if not in the tree.
anch_pq = pathlib.Path(LEGS).parent / "anchors_x12.parquet"
anch = pd.read_parquet(anch_pq)
p60 = leg_pool("a29pass_*.result.parquet").set_index(["task", "phrase"]).succ
p12 = anch[anch.arm == "passthrough"].groupby(
    ["task", "phrase"]).success.mean() * 100
base_rates["adv"] = pd.concat([p60, p12])
base_rates["adv"].index.names = ["task", "base"]

base_rates["orig"] = anch[anch.arm == "originals"].groupby(
    ["task", "phrase"]).success.mean() * 100
base_rates["orig"].index.names = ["task", "base"]

# ---- cells ------------------------------------------------------------------
A2 = {"claude": "cl", "gemini": "ge", "qwen": "qw"}
A3 = {"claude": "cla", "gemini": "gea", "qwen": "qwa"}
CONDS = ["nat", "adv", "orig"]
OUT = {}
for cond in CONDS:
    cl = {"nat": "n", "adv": "a", "orig": "o"}[cond]
    for ap_name in A2:
        arms = {}
        arm_draws = {}
        for diet in ["s", "b", "t"]:
            draws = []
            # r1
            if cond == "nat":
                draws.append(per_base(f"d34{diet}{A2[ap_name]}n_*.result.parquet"
                                      if False else f"d34{diet}{ {'claude':'cln','gemini':'gen','qwen':'qwn'}[ap_name] }_*.result.parquet",
                                      f"ph_a34_{diet}_{ap_name}_nat.parquet"))
            else:
                draws.append(per_base(f"b31{diet}{A2[ap_name]}{cl}_*.result.parquet",
                                      f"ph_a31_{diet}_{ap_name}_{cond}.parquet"))
            # r2, r3
            for dr in ["2", "3"]:
                pat = f"f36{diet}{dr}{ {'claude':'cl','gemini':'ge','qwen':'qw'}[ap_name] }{ {'nat':'n','adv':'a','orig':'o'}[cond] }_*.result.parquet"
                pq = f"ph_a36_{diet}{dr}_{ap_name}_{cond if cond != 'nat' else 'nat'}.parquet"
                draws.append(per_base(pat, pq))
            arms[diet] = pd.concat(draws, axis=1).mean(axis=1)
            arm_draws[diet] = draws
        # scaffold (r1 only; not run for orig)
        if cond == "nat":
            arms["sc"] = per_base(f"d34sc{ {'claude':'cln','gemini':'gen','qwen':'qwn'}[ap_name] }_*.result.parquet",
                                  f"ph_a34_sc_{ap_name}_nat.parquet")
        elif cond == "adv":
            arms["sc"] = per_base(f"a30{A2[ap_name]}adv_*.result.parquet",
                                  f"ph_a29_{ap_name}sc_adv.parquet")
        # draw-level inference: the rulebook draw as the unit. Each draw's
        # base-weighted delta vs baseline is one observation; a t test on
        # the n_draw=3 values asks whether the DISTILLATION PROCEDURE (not
        # just these particular books) beats baseline. df=2: only normality
        # of draw effects makes this testable at all — labeled as such.
        for diet, draws_list in arm_draws.items():
            ds = []
            for cell_d in draws_list:
                jj = pd.concat([cell_d.rename("c"),
                                base_rates[cond].rename("b")],
                               axis=1, join="inner")
                ds.append(float((jj.c - jj.b).mean()))
            m = float(np.mean(ds)); sd = float(np.std(ds, ddof=1))
            se = sd / math.sqrt(len(ds))
            t = m / se if se > 0 else 0.0
            OUT[f"{cond}|{ap_name}|{diet}_drawlevel"] = {
                "per_draw": [round(x, 2) for x in ds],
                "mean": round(m, 2), "sd_draws": round(sd, 2),
                "p_t_df2": round(_t_sf(abs(t), len(ds) - 1), 4)}
        for diet, cell in arms.items():
            baselines = {"": base_rates[cond]}
            if diet != "sc" and "sc" in arms:      # vs same-applier scaffold
                baselines["_vs_sc"] = arms["sc"]
            for suffix, base in baselines.items():
                j = pd.concat([cell.rename("c"), base.rename("b")],
                              axis=1, join="inner")
                n_expect = {"nat": 186, "adv": 72, "orig": 12}[cond]
                assert len(j) >= n_expect - 6, \
                    f"{cond}|{ap_name}|{diet}{suffix}: only {len(j)} paired bases"
                d = (j.c - j.b).values
                OUT[f"{cond}|{ap_name}|{diet}{suffix}"] = {
                    "delta": round(float(d.mean()), 2),
                    "p_perm": round(sign_flip_p(d), 4),
                    "p_t": round(paired_t_p(d), 4),
                    "n_base": len(d)}


# ---- human naturals (A39): same key shapes, data from repo job results ----
import glob as _glob
_jd = R / "results/rules_runs/r1_sim/jobs"
_hL = pd.concat([pd.read_parquet(f) for f in _glob.glob(str(_jd / "b39roll_*.result.parquet"))],
                ignore_index=True)
_hrates = _hL.groupby(["task", "phrase"]).agg(succ=("gt_success", "mean"))
_ph = pd.read_parquet(R / "results/human_naturals/a39_human_phrases.parquet")
_raw = _ph.drop_duplicates(["task", "phrase"]).merge(_hrates, on=["task", "phrase"], how="left")
_raw = _raw.dropna(subset=["succ"])
base_rates["hum"] = _raw.set_index(["task", "phrase"]).succ
base_rates["hum"].index.names = ["task", "base"]
HUM_BOOKS = {"s": ["s", "s2", "s3"], "b": ["b", "b2", "b3"], "t": ["t", "t2", "t3"]}


def _hum_map(book, ap):
    if ap != "qwen":
        f = R / f"results/human_naturals/ph_a39_{book}_{ap}.parquet"
        return pd.read_parquet(f)[["task", "base", "phrase"]]
    hit = _glob.glob(str(_jd / f"a39{book}qw_*.result.parquet"))[0]
    d = pd.read_parquet(hit).rename(columns={"phrase": "base", "rewrite": "phrase"})
    d["phrase"] = d.phrase.fillna("").astype(str)
    d.loc[d.phrase.str.strip() == "", "phrase"] = d.base
    return d[["task", "base", "phrase"]]


def _hum_base_series(book, ap):
    m = _hum_map(book, ap).merge(_hrates, on=["task", "phrase"], how="left")
    return m.groupby(["task", "base"]).succ.mean()


for ap_name in A2:
    arms = {}
    arm_draws = {}
    for diet in ["s", "b", "t"]:
        ds = [_hum_base_series(bk, ap_name) for bk in HUM_BOOKS[diet]]
        arms[diet] = pd.concat(ds, axis=1).mean(axis=1)
        arm_draws[diet] = ds
    arms["sc"] = _hum_base_series("sc", ap_name)
    for diet, draws_list in arm_draws.items():
        ds = []
        for cell_d in draws_list:
            jj = pd.concat([cell_d.rename("c"), base_rates["hum"].rename("b")],
                           axis=1, join="inner")
            ds.append(float((jj.c - jj.b).mean()))
        m = float(np.mean(ds)); sd = float(np.std(ds, ddof=1))
        se = sd / math.sqrt(len(ds)); t = m / se if se > 0 else 0.0
        OUT[f"hum|{ap_name}|{diet}_drawlevel"] = {
            "per_draw": [round(x, 2) for x in ds], "mean": round(m, 2),
            "sd_draws": round(sd, 2), "p_t_df2": round(_t_sf(abs(t), len(ds) - 1), 4)}
    for diet, cell in arms.items():
        baselines = {"": base_rates["hum"]}
        if diet != "sc":
            baselines["_vs_sc"] = arms["sc"]
        for suffix, base in baselines.items():
            j = pd.concat([cell.rename("c"), base.rename("b")], axis=1, join="inner")
            d = (j.c - j.b).dropna().values
            OUT[f"hum|{ap_name}|{diet}{suffix}"] = {
                "delta": round(float(d.mean()), 2),
                "p_perm": round(sign_flip_p(d), 4),
                "p_t": round(paired_t_p(d), 4), "n_base": len(d)}

out = R / "results/analysis/dashboard_pscores.json"
out.write_text(json.dumps(OUT, indent=1))
print("json ->", out)
for k, v in OUT.items():
    if "delta" in v:
        print(f"{k:22s} delta={v['delta']:+6.2f}  p_perm={v['p_perm']:.4f} "
              f"p_t={v['p_t']:.4f}  n={v['n_base']}")
    else:
        print(f"{k:26s} draws={v['per_draw']}  p_t(df=2)={v['p_t_df2']:.4f}")

# ---- simple chart: delta + p grid ------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RL = {"sc": "no rules", "s": "out-of-finetune", "b": "both",
      "t": "in-finetune"}
CN = {"adv": "Adversarial\n(n=72)", "nat": "LLM-Generated Naturals\n(n=186)",
      "hum": "Human-Generated Naturals\n(n=363)", "orig": "Original\n(n=12)"}


PAPER = os.environ.get("PAPER") == "1"   # paper variant: no in-image title


def p_grid(rows, suffix, title, png_name, conds=("adv", "hum", "nat")):
    if PAPER:
        title, png_name = None, png_name.replace(".png", "_paper.png")
    fig, ax = plt.subplots(figsize=(9.6, 0.62 * len(rows) + 1.6))
    ax.set_xlim(0, 3); ax.set_ylim(0, len(rows)); ax.invert_yaxis()
    ax.axis("off")
    for ci, cond in enumerate(conds):
        ax.text(ci + 0.5, -0.55, CN[cond], ha="center", va="center",
                fontsize=10, fontweight="bold")
        for ri, (ap_name, diet) in enumerate(rows):
            v = OUT.get(f"{cond}|{ap_name}|{diet}{suffix}")
            if v is None:
                ax.text(ci + 0.5, ri + 0.55, "no scaffold arm" if suffix
                        else "not run", ha="center", fontsize=8,
                        color="#a0aec0")
                continue
            p = v["p_perm"]
            fc = ("#c6f6d5" if v["delta"] > 0 else "#fed7d7") if p < 0.05 \
                else "#edf2f7"
            ax.add_patch(plt.Rectangle((ci + 0.02, ri + 0.06), 0.96, 0.88,
                                       fc=fc, ec="#cbd5e0", lw=0.7))
            ps = f"p={p:.4f}" if p >= 1e-4 else "p<0.0001"
            ax.text(ci + 0.5, ri + 0.52,
                    f"{v['delta']:+.1f} pp   {ps}", ha="center", va="center",
                    fontsize=9.5,
                    fontweight="bold" if p < 0.05 else "normal")
    for ri, (ap_name, diet) in enumerate(rows):
        ax.text(-0.04, ri + 0.52, f"{ap_name} · {RL[diet]}", ha="right",
                va="center", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10.5, pad=46)
    fig.tight_layout()
    png = R / f"results/analysis/{png_name}"
    fig.savefig(png, dpi=150, bbox_inches="tight", pad_inches=0.25)
    print("chart ->", png)


p_grid([(ap, d) for ap in ["claude", "gemini", "qwen"]
        for d in ["sc", "s", "b", "t"]], "",
       "Dashboard bars vs no-rephraser — paired-by-base sign-flip "
       "permutation p\n(cells pooled over draws; green/red = significant "
       "gain/harm at p<0.05, grey = not significant)",
       "dashboard_pscores.png")
p_grid([(ap, d) for ap in ["claude", "gemini", "qwen"]
        for d in ["s", "b", "t"]], "_vs_sc",
       "Rulebook cells vs the SAME APPLIER's no-rules scaffold — "
       "paired-by-base permutation p\n(what the rules add beyond the "
       "rephrase scaffold; original has no scaffold arm)",
       "dashboard_pscores_vs_scaffold.png")
