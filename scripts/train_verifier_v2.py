#!/usr/bin/env python3
"""Verifier-v2 residual head, rung 1: frozen features, learned aggregation.

Architecture (HANDOFF 1b): Deep-Sets over per-frame (GT, measured) evidence --
shared per-frame MLP -> masked mean over frames -> episode MLP -> masked mean
over episodes -> head. Deployed score is a RESIDUAL on the frozen proxy:

    f(phrase) = alpha * [0.4445*mean(z) + 11.3193*(-mean(grip))] + g(frames)

alpha init 1, g's output layer zero-init, so at initialization f IS the
current reward, bit for bit. Trained with within-task RankNet on confident
pairs (never success-rate regression -- the pooled-refit lesson: regression
chases per-task base rates and kills ranking). One scalar per phrase at
inference; sort. No calibration head.

This rung trains on what exists locally: fine_exam per-frame features
(z_row, grip_row, t) for 254 phrases / 8 tasks. Rich 64-d embeddings for all
434 phrases need scripts/extract_v2_features.py on a pod; this file's data
loader gains an --features flag for that parquet when it lands.

Honest evaluation: rotating task folds (2 test / 1 early-stop / 5 train).
The reported number is agreement on tasks the model never saw.

RUNG-1 RESULT (2026-08-11, thin features): no transferable gain. Pooled
held-out-task agreement 78.2 (base) vs 75.2-77.5 (v2) across alpha/penalty
configs; every fold where the residual grew (resid_sd ~0.5-1.2) LOST accuracy
on its test tasks, and a 1-task early-stop set cannot certify transfer at this
scale. Consistent with the rank-fit result: two scalars per frame carry
nothing the frozen mean does not already extract. The trainer is validated
machinery awaiting rich inputs -- run scripts/extract_v2_features.py on a pod
(e6 for all 15 tasks) and retrain with --features on the 64-d embeddings and
the 434-phrase set before drawing any further conclusion.

  .venv/bin/python scripts/train_verifier_v2.py
"""
import argparse
import itertools
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BZ, BG = 0.4445, 11.3193      # frozen calibrated-logit coefficients
CONF, MIN_GAP = 0.8, 5.0


# ---------------------------------------------------------------- data

def load_phrases_rich(path):
    """Rich per-frame features from extract_v2_features.py: slot 0 = mean member
    logit (the deployed z), slot 1 = grip, slot 2 = t_frac, then the 64-d
    penultimate embedding and the per-member logits. Ground truth = the full
    434-phrase set (fine_exam gt_n per panel; nat/adv at n=36)."""
    feats = pd.read_parquet(path)
    fe = pd.read_parquet("results/analysis/fine_exam_phrases.parquet")
    gt = {(r.task, r.phrase): (float(r.gt_success), int(r.gt_n)) for r in fe.itertuples()}
    for f in ("results/analysis/sim_rollouts_natadv_0of2.parquet",
              "results/analysis/sim_rollouts_natadv_1of2.parquet"):
        for r in pd.read_parquet(f).itertuples():
            gt.setdefault((r.task, r.phrase), (float(r.gt_success), 36))

    phrases = []
    for (task, phrase), g in feats.groupby(["task", "phrase"]):
        if (task, phrase) not in gt:
            continue
        eps = []
        for _, ge in g.groupby("episode_index"):
            ge = ge.sort_values("t")
            tmax = max(float(ge.t.max()), 1.0)
            emb = np.stack(ge.embed.map(np.asarray))            # (F, 64)
            ml = np.stack(ge.member_logits.map(np.asarray))     # (F, M)
            z = ml.mean(axis=1, keepdims=True)                  # deployed z per frame
            grip = ge.grip_row.values[:, None]
            tf = (ge.t.values / tmax)[:, None]
            eps.append(np.concatenate([z, grip, tf, emb, ml], axis=1).astype(np.float32))
        s, n = gt[(task, phrase)]
        phrases.append({"task": task, "phrase": phrase, "eps": eps, "gt": s, "gt_n": n})
    return phrases


def load_phrases():
    feats = pd.concat([pd.read_parquet("results/analysis/fine_exam_features_native.parquet"),
                       pd.read_parquet("results/analysis/fine_exam_features_oov.parquet")],
                      ignore_index=True)
    pan = pd.read_parquet("results/analysis/fine_exam_phrases.parquet")
    gt = {(r.task, r.phrase): (float(r.gt_success), int(r.gt_n)) for r in pan.itertuples()}

    phrases = []
    for (task, phrase), g in feats.groupby(["task", "phrase"]):
        if (task, phrase) not in gt:
            continue
        eps = []
        for _, ge in g.groupby("episode_index"):
            ge = ge.sort_values("t")
            tmax = max(float(ge.t.max()), 1.0)
            eps.append(np.stack([ge.z_row.values, ge.grip_row.values,
                                 ge.t.values / tmax], axis=1).astype(np.float32))
        s, n = gt[(task, phrase)]
        phrases.append({"task": task, "phrase": phrase, "eps": eps,
                        "gt": s, "gt_n": n})
    return phrases


def pad_tensors(phrases):
    """-> x (P,E,F,D) raw features, mask_f (P,E,F), mask_e (P,E).
    Slot 0 is always the deployed z, slot 1 grip, slot 2 t_frac; rich features
    append the embedding + member logits after."""
    E = max(len(p["eps"]) for p in phrases)
    F = max(max(len(e) for e in p["eps"]) for p in phrases)
    P, D = len(phrases), phrases[0]["eps"][0].shape[1]
    x = np.zeros((P, E, F, D), dtype=np.float32)
    mf = np.zeros((P, E, F), dtype=np.float32)
    me = np.zeros((P, E), dtype=np.float32)
    for i, p in enumerate(phrases):
        for j, e in enumerate(p["eps"]):
            x[i, j, :len(e)] = e
            mf[i, j, :len(e)] = 1.0
            me[i, j] = 1.0
    return torch.from_numpy(x), torch.from_numpy(mf), torch.from_numpy(me)


def confident_pairs(phrases, tasks):
    """[(i_better, j_worse)] within-task, ordering-confidence >= CONF."""
    idx = {t: [i for i, p in enumerate(phrases) if p["task"] == t] for t in tasks}
    prs = []
    for t in tasks:
        for i, j in itertools.combinations(idx[t], 2):
            a, b = phrases[i], phrases[j]
            gap = abs(a["gt"] - b["gt"])
            se = 100 * math.sqrt(a["gt"] / 100 * (1 - a["gt"] / 100) / a["gt_n"]
                                 + b["gt"] / 100 * (1 - b["gt"] / 100) / b["gt_n"])
            conf = 0.5 * (1 + math.erf((gap / se) / math.sqrt(2))) if se > 0 else 1.0
            if gap >= MIN_GAP and conf >= CONF:
                prs.append((i, j) if a["gt"] > b["gt"] else (j, i))
    return prs


# ---------------------------------------------------------------- model

class V2Head(nn.Module):
    def __init__(self, d_in=3, h=48, mu=None, sd=None):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d_in) if mu is None else mu)
        self.register_buffer("sd", torch.ones(d_in) if sd is None else sd)
        self.phi = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.ep = nn.Sequential(nn.Linear(h, h), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        nn.init.zeros_(self.head[-1].weight)     # residual starts at exactly 0
        nn.init.zeros_(self.head[-1].bias)
        # alpha frozen at 1 by default: on thin features a trainable alpha let
        # one fold trade the (good) base away for a residual that didn't
        # transfer (87.1 -> 80.4 on its held-out tasks). Unfreeze only when
        # rich features give the residual something real to say.
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x, mf, me, noise=0.0):
        mf3, me3 = mf.unsqueeze(-1), me.unsqueeze(-1)
        # frozen-proxy base from the SAME (possibly subsampled) evidence
        nf = mf3.sum(dim=(1, 2)).clamp(min=1.0)
        z_mean = (x[..., 0:1] * mf3).sum(dim=(1, 2)) / nf
        g_mean = (x[..., 1:2] * mf3).sum(dim=(1, 2)) / nf
        base = (BZ * z_mean - BG * g_mean).squeeze(-1)

        xs = (x - self.mu) / self.sd
        if noise > 0:
            xs = xs + noise * torch.randn_like(xs)
        h = self.phi(xs)
        epm = (h * mf3).sum(2) / mf3.sum(2).clamp(min=1.0)
        he = self.ep(epm)
        ctx = (he * me3).sum(1) / me3.sum(1).clamp(min=1.0)
        return self.alpha * base + self.head(ctx).squeeze(-1), base


def subsample_masks(phrases, mf, me, rng):
    """Random evidence-dropout: keep 2..E episodes, 2..F frames each."""
    mf2, me2 = mf.clone(), me.clone()
    for i, p in enumerate(phrases):
        E = len(p["eps"])
        if E >= 3:
            drop = rng.choice(E, size=E - rng.integers(2, E + 1), replace=False)
            me2[i, drop] = 0.0
            mf2[i, drop] = 0.0
        for j, e in enumerate(p["eps"]):
            if me2[i, j] == 0 or len(e) < 3:
                continue
            drop = rng.choice(len(e), size=len(e) - rng.integers(2, len(e) + 1),
                              replace=False)
            mf2[i, j, drop] = 0.0
    return mf2, me2


# ---------------------------------------------------------------- train/eval

def agreement(scores, prs):
    if not prs:
        return float("nan")
    ok = sum(scores[i] > scores[j] for i, j in prs)
    return 100 * ok / len(prs)


def run_fold(phrases, x, mf, me, train_t, stop_t, test_t, args, rng):
    tr_prs = confident_pairs(phrases, train_t)
    st_prs = confident_pairs(phrases, stop_t)
    te_prs = confident_pairs(phrases, test_t)

    tr_idx = [i for i, p in enumerate(phrases) if p["task"] in train_t]
    D = x.shape[-1]
    mu = x[tr_idx].reshape(-1, D)[mf[tr_idx].reshape(-1) > 0].mean(0)
    sd = x[tr_idx].reshape(-1, D)[mf[tr_idx].reshape(-1) > 0].std(0) + 1e-6
    model = V2Head(d_in=D, mu=mu, sd=sd)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ib = torch.tensor([i for i, _ in tr_prs])
    iw = torch.tensor([j for _, j in tr_prs])
    best = (-1.0, 0, None)
    for step in range(1, args.steps + 1):
        model.train()
        mf2, me2 = subsample_masks(phrases, mf, me, rng)
        f, b = model(x, mf2, me2, noise=args.noise)
        loss = (torch.nn.functional.softplus(-(f[ib] - f[iw])).mean()
                + args.resid_penalty * (f - model.alpha * b).pow(2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            model.eval()
            with torch.no_grad():
                fe, _ = model(x, mf, me)
            acc = agreement(fe.numpy(), st_prs)
            if acc > best[0]:
                best = (acc, step, {k: v.clone() for k, v in model.state_dict().items()})
            elif step - best[1] >= 20 * args.patience:
                break
    if best[2] is not None:
        model.load_state_dict(best[2])
    model.eval()
    with torch.no_grad():
        fe, base = model(x, mf, me)
    return {"test_v2": round(agreement(fe.numpy(), te_prs), 1),
            "test_base": round(agreement(base.numpy(), te_prs), 1),
            "n_test_pairs": len(te_prs), "best_step": best[1],
            "alpha": round(model.alpha.item(), 3),
            "resid_sd": round(float(fe.sub(model.alpha * base).std()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--resid-penalty", type=float, default=1e-2,
                    help="L2 on the residual's magnitude: the head must earn "
                         "every unit it moves the score away from the base")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", default="",
                    help="rich per-frame parquet from extract_v2_features.py; "
                         "default = thin fine_exam features")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    phrases = load_phrases_rich(args.features) if args.features else load_phrases()
    x, mf, me = pad_tensors(phrases)
    tasks = sorted({p["task"] for p in phrases})
    print(f"{len(phrases)} phrases, {len(tasks)} tasks, tensor {tuple(x.shape)}")

    # rotating folds: ~1/3 of tasks test, 2 stop, rest train (15 -> 5/2/8,
    # the 10/5 split; 8 thin-feature tasks -> 2/1/5 as before)
    order = list(rng.permutation(tasks))
    n_test = max(2, round(len(order) / 3))
    n_stop = 2 if len(order) >= 12 else 1
    folds, out = [], {"folds": []}
    for k in range(0, len(order) - (len(order) % n_test or 0), n_test):
        test_t = order[k:k + n_test]
        if not test_t:
            continue
        rest = [t for t in order if t not in test_t]
        stop_t = rest[:n_stop]
        train_t = rest[n_stop:]
        folds.append((train_t, stop_t, test_t))

    wsum = vsum = bsum = 0
    for train_t, stop_t, test_t in folds:
        r = run_fold(phrases, x, mf, me, train_t, stop_t, test_t, args, rng)
        r["test_tasks"] = [t.replace("widowx_", "") for t in test_t]
        out["folds"].append(r)
        print(f"fold {r['test_tasks']}: base {r['test_base']}  ->  v2 {r['test_v2']} "
              f"({r['n_test_pairs']} pairs, alpha {r['alpha']}, resid_sd {r['resid_sd']}, "
              f"step {r['best_step']})")
        wsum += r["n_test_pairs"]
        vsum += r["test_v2"] * r["n_test_pairs"]
        bsum += r["test_base"] * r["n_test_pairs"]
    out["pooled"] = {"base": round(bsum / wsum, 1), "v2": round(vsum / wsum, 1),
                     "n_pairs": wsum}
    print(f"\nPOOLED held-out-task agreement: base {out['pooled']['base']} "
          f"-> v2 {out['pooled']['v2']}  ({wsum} pairs)")

    # deployment prototype: same recipe on all tasks, median best_step, no peeking
    steps = int(np.median([f["best_step"] for f in out["folds"]]) or 200)
    D = x.shape[-1]
    mu = x.reshape(-1, D)[mf.reshape(-1) > 0].mean(0)
    sd = x.reshape(-1, D)[mf.reshape(-1) > 0].std(0) + 1e-6
    model = V2Head(d_in=D, mu=mu, sd=sd)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    prs = confident_pairs(phrases, tasks)
    ib = torch.tensor([i for i, _ in prs]); iw = torch.tensor([j for _, j in prs])
    for _ in range(steps):
        mf2, me2 = subsample_masks(phrases, mf, me, rng)
        f, _ = model(x, mf2, me2, noise=args.noise)
        torch.nn.functional.softplus(-(f[ib] - f[iw])).mean().backward()
        opt.step(); opt.zero_grad()
    torch.save({"state_dict": model.state_dict(), "steps": steps,
                "note": "rung-1 prototype, thin features (z,grip,t_frac); "
                        "cv numbers in results/analysis/verifier_v2_proto.json"},
               "results/checkpoints/verifier_v2_head_proto.pt")
    out["final_steps"] = steps
    with open("results/analysis/verifier_v2_proto.json", "w") as fjson:
        json.dump(out, fjson, indent=2)
    print("saved results/checkpoints/verifier_v2_head_proto.pt + verifier_v2_proto.json")


if __name__ == "__main__":
    main()
