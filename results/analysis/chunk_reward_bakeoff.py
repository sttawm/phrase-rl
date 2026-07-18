"""v7 offline reward bake-off: chunk-error-only small MLP vs gripper-only vs hybrids.

User spec (2026-07-18): train a SMALL MLP (32x16, plus a logistic floor) on
CHUNK-ERROR features ONLY -- decoded-chunk diffs vs a_star (norm_l2, grip_err,
per-DoF / per-timestep error summaries), NO flow features, NO raw absolute
actions (the domain-carrying channels), hard negatives kept. Evaluate offline on
the existing labeled tables with the same protocol as the dawn re-analysis
(scratchpad/reanalysis.py): fine = pairwise acc (success gap > 10pp) + spearman
on own-16; coarse = relevance AUC (own vs cross) + red-team mean rank.

Datasets:
  bridge-val   : verifier_features_val_multit.parquet (250 held-out episodes x 4
                 quartile frames, episode-disjoint from train) -- coarse AUC
                 overall / hard / easy. This is the TRAINING-domain gate metric.
  eggplant_sim : 60 recorded sim trajectories x 35 phrases (the leave-DOMAIN-out
                 table where the locked ensemble inverts fine ranking).
  study_multit : 4-task quartile-context study table (the 4f decision table,
                 real Bridge frames).
  study_oldctx : old single-frame study contexts (full spoon labels).

Training rows: verifier_features_train_multit.parquet (2,000 Bridge episodes x 4
quartile frames x {own, hard, easy}).  NOTE the episode-overlap caveat printed at
runtime: study_multit contexts may reuse train episodes (same caveat applies to
the locked ensemble; eggplant_sim and bridge-val are fully disjoint).

Run:  .venv/bin/python results/analysis/chunk_reward_bakeoff.py
Out:  results/analysis/chunk_reward_bakeoff.json
"""

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

REPO = "/Users/sttawm/dev/robotics/phrase-rl"
os.chdir(REPO)
sys.path.insert(0, f"{REPO}/src")
RAW = "results/overnight/raw"
K_FLOW, K_DEC, H, D = 8, 4, 4, 7

from phrase_rl.train_verifier import MLP, auc                      # noqa: E402
from phrase_rl.verifier_reward import VerifierEnsemble             # noqa: E402
from phrase_rl.eval_verifier_study import study_feature_rows, load_success  # noqa: E402

# per-dim action std from verifier_features_train_meta.json (dataset stats used
# by norm_l2); used to normalize per-DoF error summaries.
DIM_STD = np.array(json.load(open(f"{RAW}/verifier_features_train_meta.json"))["dim_std"],
                   dtype=np.float32)  # (7,)


# ---------------- chunk-error featurization (NO flow, NO raw actions) ----------------

def _stack(col):
    return np.stack([np.asarray(x, dtype=np.float32) for x in col])


def chunk_features(df, full: bool) -> np.ndarray:
    dec = _stack(df.decoded).reshape(-1, K_DEC, H, D)
    a = _stack(df.a_star).reshape(-1, 1, H, D)
    err = (dec - a) / DIM_STD[None, None, None, :]        # normalized error
    err_md = err.mean(axis=1)                             # (N, H, D) mean over draws
    parts = [
        _stack(df.norm_l2),                               # (N, Kd)   per-draw chunk L2
        _stack(df.grip_err),                              # (N, Kd)   per-draw gripper err
        np.sqrt((err_md ** 2).mean(axis=2)),              # (N, H)    per-timestep RMS err
        np.abs(err_md).mean(axis=1),                      # (N, D)    per-DoF mean abs err
    ]
    if full:
        parts += [
            err_md.reshape(len(df), -1),                  # (N, H*D)  signed mean error
            dec.std(axis=1).reshape(len(df), -1),         # (N, H*D)  decode spread
        ]
    return np.concatenate(parts, axis=1).astype(np.float32)


# ---------------- training (mirrors train_verifier.py, smaller net) ----------------

def train_chunk_model(Xtr, ytr, Xva, yva, va_roles, hidden, seed, epochs=400,
                      patience=40, lr=1e-3, wd=1e-4):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    model = MLP(Xtr.shape[1], hidden=hidden) if hidden else nn.Linear(Xtr.shape[1], 1)
    fwd = (lambda x: model(x)) if hidden else (lambda x: model(x).squeeze(-1))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / ytr.sum()))
    Xt, yt, Xv = torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(Xva)
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            lossf(fwd(Xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            a = auc(yva, fwd(Xv).numpy())
        if a > best_auc:
            best_auc, best_state, bad = a, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        sv = fwd(Xv).numpy()
    hard = (va_roles == "hard") | (yva == 1)
    easy = (va_roles == "easy") | (yva == 1)
    stats = {"val_auc": round(auc(yva, sv), 4),
             "val_auc_hard": round(auc(yva[hard], sv[hard]), 4),
             "val_auc_easy": round(auc(yva[easy], sv[easy]), 4)}
    return model, fwd, mu, sd, stats


class ChunkScorer:
    """Mean logit over seed-ensemble members on chunk-error features."""

    def __init__(self, members, full):
        self.members, self.full = members, full

    @torch.no_grad()
    def scores(self, feats_df) -> pd.Series:
        X = chunk_features(feats_df, self.full)
        zs = []
        for model, fwd, mu, sd in self.members:
            zs.append(fwd(torch.from_numpy((X - mu) / sd)).numpy())
        z = np.mean(zs, axis=0)
        return feats_df.assign(score=z).groupby(["task", "phrase"]).score.mean()


# ---------------- eval-dataset builders (from reanalysis.py) ----------------

def rows_from_tables(verbose_pq, l2_pq, a_star_map):
    vb = pd.read_parquet(verbose_pq)
    l2 = pd.read_parquet(l2_pq)
    rows = []
    l2g = {k: g for k, g in l2.groupby(["task", "phrase", "episode_index", "t"])}
    for key, g in vb.groupby(["task", "phrase", "episode_index", "t"]):
        t, p, e, ts = key
        g = g[g.k < K_FLOW].sort_values("k")
        h = l2g.get(key)
        if h is None or len(g) < K_FLOW:
            continue
        h = h[h.k < K_DEC].sort_values("k")
        if len(h) < K_DEC:
            continue
        a = a_star_map.get((t, e, ts), a_star_map.get((e, ts)))
        if a is None:
            continue
        rows.append({
            "task": t, "phrase": p, "episode_index": e, "t": ts,
            "flow_loss": g.loss.values.astype(np.float32),
            "flow_v": np.concatenate([np.asarray(x, np.float32) for x in g.v]),
            "flow_u": np.concatenate([np.asarray(x, np.float32) for x in g.u]),
            "decoded": np.concatenate([np.asarray(x, np.float32) for x in h.decoded]),
            "norm_l2": h.norm_l2.values.astype(np.float32),
            "grip_err": h.grip_err.values.astype(np.float32),
            "a_star": np.asarray(a, np.float32),
        })
    return pd.DataFrame(rows)


def hybrid_rel_scores(gate_sc, fine_sc):
    """Relative coarse gate (GRPO-group analog): within each task, phrases at or
    above the median gate score pass; passers ranked by fine, failers by gate."""
    out = {}
    for t in {t for t, _ in gate_sc.index}:
        sub = gate_sc[[k for k in gate_sc.index if k[0] == t]]
        med = float(np.median(sub.values))
        for k in sub.index:
            out[k] = (1000.0 + fine_sc[k]) if gate_sc[k] >= med else gate_sc[k] - 1000.0
    return pd.Series(out)


def metrics_for(scores, phr, succ, tasks):
    per_task, pooled_pairs, aucs, sps = {}, [], [], []
    for t in tasks:
        sub = phr[phr.task == t]
        own = [p for p in sub[sub.arm == "own"].phrase if (t, p) in scores.index]
        cross = [p for p in sub[sub.arm == "cross"].phrase if (t, p) in scores.index]
        rts = [p for p in sub[sub.arm.isin(["redteam1", "redteam2"])].phrase if (t, p) in scores.index]
        rolled = [p for p in own if (t, p) in succ]
        m = {}
        if len(rolled) >= 8:
            sc = np.array([scores[(t, p)] for p in rolled])
            su = np.array([succ[(t, p)] for p in rolled])
            m["top1_regret"] = round(float(su.max() - su[np.argmax(sc)]), 1)
            m["spearman_own"] = round(float(spearmanr(sc, su)[0]), 3)
            sps.append(m["spearman_own"])
            pairs = [(scores[(t, a)] - scores[(t, b)], succ[(t, a)] - succ[(t, b)])
                     for a, b in itertools.combinations(rolled, 2)
                     if abs(succ[(t, a)] - succ[(t, b)]) > 10]
            if pairs:
                m["pairwise_acc"] = round(float(np.mean([(a > 0) == (b > 0) for a, b in pairs])), 3)
                m["n_pairs"] = len(pairs)
                pooled_pairs += pairs
        if own and cross:
            y = np.array([1] * len(own) + [0] * len(cross))
            s = np.array([scores[(t, p)] for p in own + cross])
            m["relevance_auc"] = round(auc(y, s), 3)
            aucs.append(m["relevance_auc"])
        if rts and own:
            allp = own + rts
            s = np.array([scores[(t, p)] for p in allp])
            ranks = len(allp) + 1 - pd.Series(s).rank().values
            m["rt_mean_rank"] = round(float(np.mean([ranks[allp.index(p)] for p in rts])), 1)
        per_task[t] = m
    pooled = {}
    if pooled_pairs:
        pooled["pairwise_acc"] = round(float(np.mean([(a > 0) == (b > 0) for a, b in pooled_pairs])), 3)
        pooled["n_pairs"] = len(pooled_pairs)
    if aucs:
        pooled["mean_relevance_auc"] = round(float(np.mean(aucs)), 3)
    if sps:
        pooled["mean_spearman_own"] = round(float(np.mean(sps)), 3)
    return {"per_task": per_task, "pooled": pooled}


def main():
    tr = pd.read_parquet(f"{RAW}/verifier_features_train_multit.parquet")
    va = pd.read_parquet(f"{RAW}/verifier_features_val_multit.parquet")
    ytr = tr.label.values.astype(np.float32)
    yva = va.label.values.astype(np.float32)

    report = {"train_rows": len(tr), "val_rows": len(va),
              "role_counts": tr.role.value_counts().to_dict()}

    # -------- train chunk-only models: MLP 32x16 (min + full feats) + logistic floor
    models = {}
    for name, full, hidden in [("chunk_mlp_min", False, (32, 16)),
                               ("chunk_mlp_full", True, (32, 16)),
                               ("chunk_logistic_min", False, None),
                               ("chunk_logistic_full", True, None)]:
        Xtr, Xva = chunk_features(tr, full), chunk_features(va, full)
        members, stats_all = [], []
        seeds = [0, 1, 2, 3, 4] if hidden else [0]
        for seed in seeds:
            model, fwd, mu, sd, stats = train_chunk_model(
                Xtr, ytr, Xva, yva, va.role.values, hidden, seed)
            members.append((model, fwd, mu, sd))
            stats_all.append(stats)
        agg = {k: round(float(np.mean([s[k] for s in stats_all])), 4) for k in stats_all[0]}
        agg["d_in"] = int(Xtr.shape[1])
        agg["n_seeds"] = len(seeds)
        # ensemble-level bridge-val AUC (mean logit)
        sc = ChunkScorer(members, full)
        with torch.no_grad():
            z = np.mean([f(torch.from_numpy((Xva - m) / s)).numpy()
                         for _, f, m, s in members], axis=0)
        hard = (va.role.values == "hard") | (yva == 1)
        agg["ens_val_auc"] = round(auc(yva, z), 4)
        agg["ens_val_auc_hard"] = round(auc(yva[hard], z[hard]), 4)
        models[name] = sc
        report.setdefault("bridge_val", {})[name] = agg
        print(name, agg)

    # -------- eval datasets
    ens4 = VerifierEnsemble("results/checkpoints/verifier_reward_ensemble_4f.json")
    phr = pd.read_parquet("results/phrase_artifacts/study_phrases_all.parquet")
    succ = load_success()
    datasets = {}
    sc_ctx = pd.read_parquet(f"{RAW}/sim_contexts_eggplant.parquet")
    amap = {(r.task, r.episode_index, r.t): np.asarray(r.action_chunk, np.float32)
            for r in sc_ctx.itertuples()}
    datasets["eggplant_sim"] = rows_from_tables(
        f"{RAW}/eggplant_sim_verbose.parquet", f"{RAW}/eggplant_sim_l2.parquet", amap)
    ast = pd.read_parquet(f"{RAW}/multit_astar.parquet")
    amap_m = {(int(r.episode_index), int(r.t)): np.asarray(r.action_chunk, np.float32)
              for r in ast.itertuples()}
    datasets["study_multit"] = rows_from_tables(
        f"{RAW}/study_multit_verbose.parquet", f"{RAW}/study_multit_l2.parquet", amap_m)
    datasets["study_oldctx"] = study_feature_rows()

    # episode-overlap caveat
    tr_eps = set(tr.episode_index.unique())
    for name, feats in datasets.items():
        ov = set(feats.episode_index.unique()) & tr_eps
        report.setdefault("episode_overlap_with_train", {})[name] = len(ov)
        print(f"{name}: {len(feats)} rows, {len(ov)} episodes overlap train")

    for name, feats in datasets.items():
        tasks = sorted(feats.task.unique())
        e4 = feats.assign(score=ens4.member_logits(feats).mean(axis=1)) \
                  .groupby(["task", "phrase"]).score.mean()
        grip = -feats.groupby(["task", "phrase"]).grip_err.apply(
            lambda s: float(np.mean(np.stack(s.values))))
        scorers = {
            "ensemble_4f_native": e4,
            "raw_grip": grip,
            "chunk_mlp_min": models["chunk_mlp_min"].scores(feats),
            "chunk_mlp_full": models["chunk_mlp_full"].scores(feats),
            "chunk_logistic_min": models["chunk_logistic_min"].scores(feats),
            "chunk_logistic_full": models["chunk_logistic_full"].scores(feats),
        }
        scorers["hybrid_relgate4f_grip"] = hybrid_rel_scores(e4, grip)
        scorers["hybrid_relgate4f_chunkmlp"] = hybrid_rel_scores(e4, scorers["chunk_mlp_min"])
        scorers["hybrid_relgate4f_chunkmlp_full"] = hybrid_rel_scores(e4, scorers["chunk_mlp_full"])
        rep = {s: metrics_for(sc, phr, succ, tasks) for s, sc in scorers.items()}
        report[name] = rep
        print(f"\n########## {name} ##########")
        for s, r in rep.items():
            print(f"== {s} == pooled {r['pooled']}")
            for t, m in r["per_task"].items():
                print(f"  {t.replace('widowx_', ''):26s} {m}")

    out = "results/analysis/chunk_reward_bakeoff.json"
    json.dump(report, open(out, "w"), indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
