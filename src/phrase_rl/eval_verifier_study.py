"""Score trained verifiers against the study's rollout-success labels, next to the
raw baselines. Bridge-trained models never saw SIMPLER: every task is out-of-sample.

Metrics per task (and pooled): top-1 selection regret on the own-16, pairwise
preference accuracy (success gap > 10pp), Spearman (own-16 / all rolled),
relevance AUC (own vs cross), red-team placement (mean rank of RT phrases,
1 = scored best -> bad).

Baselines are computed from the same banked study parquets. The flow-mode
verifier needs raw v/u at the study contexts: results/overnight/raw/
study_verbose_flow.parquet (produced by study_verbose_rescore on pod 3);
flow/both verifiers are skipped with a notice until it exists.

  .venv/bin/python -m phrase_rl.eval_verifier_study \
    --models results/checkpoints/verifier_l2 [verifier_flow verifier_both] \
    --out results/verifier_study_eval.json
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch

from phrase_rl.train_verifier import MLP, auc
from phrase_rl.verifier_features import H, D, K_DEC, K_FLOW

RAW = "results/overnight/raw"


def load_success():
    per = {}
    for f in glob.glob(f"{RAW}/rollouts_study_*.parquet"):
        d = pd.read_parquet(f)
        for (t, p), s in d.groupby(["task", "phrase"]).success.mean().items():
            per[(t, p)] = s * 100
    return per


def study_feature_rows():
    """Per (task, phrase): one pooled feature row per context, verifier-schema-compatible."""
    l2 = pd.read_parquet(f"{RAW}/study_scores_l2_k8.parquet")
    ctx = pd.read_parquet("data/contexts_0c_match.parquet")
    a_star = {(r.task, r.episode_index, r.t): np.asarray(r.action_chunk, dtype=np.float32)
              for r in ctx.itertuples()}
    fl = pd.read_parquet(f"{RAW}/study_scores_flow_k64.parquet")
    fl = fl[fl.k < K_FLOW]  # same seed => first K draws match training slots
    vfile = f"{RAW}/study_verbose_flow.parquet"
    vu = pd.read_parquet(vfile) if os.path.exists(vfile) else None

    rows = []
    for (t, p, e, ts), g in l2.groupby(["task", "phrase", "episode_index", "t"]):
        g = g[g.k < K_DEC].sort_values("k")
        if len(g) < K_DEC:
            continue
        row = {"task": t, "phrase": p, "episode_index": e, "t": ts,
               "decoded": np.concatenate([np.asarray(x, dtype=np.float32) for x in g.decoded]),
               "norm_l2": g.norm_l2.values.astype(np.float32),
               "grip_err": g.grip_err.values.astype(np.float32),
               "a_star": a_star[(t, e, ts)]}
        fg = fl[(fl.task == t) & (fl.phrase == p) & (fl.episode_index == e) & (fl.t == ts)]
        row["flow_loss"] = fg.sort_values("k").loss.values.astype(np.float32)
        if vu is not None:
            vg = vu[(vu.task == t) & (vu.phrase == p) & (vu.episode_index == e) & (vu.t == ts)].sort_values("k")
            if len(vg) == K_FLOW:
                row["flow_v"] = np.concatenate([np.asarray(x, dtype=np.float32) for x in vg.v])
                row["flow_u"] = np.concatenate([np.asarray(x, dtype=np.float32) for x in vg.u])
        rows.append(row)
    return pd.DataFrame(rows)


def model_scores(prefix, feats):
    ck = torch.load(prefix + ".pt", weights_only=False)
    mode = ck["mode"]
    if mode.endswith("+hist"):
        h = pd.read_parquet("data/action_history.parquet")
        feats = feats.merge(h, on=["episode_index", "t"], how="left")
        if feats.t_frac.isna().any():
            print(f"SKIP {prefix}: history missing for {int(feats.t_frac.isna().sum())} study rows")
            return None
    if mode in ("flow", "both", "deploy") and "flow_v" not in feats.columns:
        print(f"SKIP {prefix}: mode={mode} needs study_verbose_flow.parquet (queued on pod 3)")
        return None
    from phrase_rl.verifier_features import featurize
    X = (featurize(feats, mode, ck.get("include_diff", True)) - ck["mu"]) / ck["sd"]
    m = MLP(ck["d_in"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    with torch.no_grad():
        s = m(torch.from_numpy(X.astype(np.float32))).numpy()
    return feats.assign(score=s).groupby(["task", "phrase"]).score.mean()


def baseline_scores(feats):
    out = {}
    g = feats.groupby(["task", "phrase"])
    out["raw_-flow_mean"] = -g.apply(lambda x: np.mean(np.stack(x.flow_loss)), include_groups=False)
    out["raw_-l2_mean"] = -g.apply(lambda x: np.mean(np.stack(x.norm_l2)), include_groups=False)
    out["raw_-grip_err"] = -g.apply(lambda x: np.mean(np.stack(x.grip_err)), include_groups=False)
    out["raw_-act_spread"] = -g.apply(
        lambda x: np.mean([np.asarray(d).reshape(K_DEC, H * D).std(0).mean() for d in x.decoded]),
        include_groups=False)
    return out


def metrics_for(scores, phr, succ):
    per_task, pooled_pairs = {}, []
    for t in phr.task.unique():
        sub = phr[phr.task == t]
        own = [p for p in sub[sub.arm == "own"].phrase if (t, p) in scores.index]
        cross = [p for p in sub[sub.arm == "cross"].phrase if (t, p) in scores.index]
        rts = [p for p in sub[sub.arm.isin(["redteam1", "redteam2"])].phrase if (t, p) in scores.index]
        rolled = [p for p in own if (t, p) in succ]
        m = {}
        if len(rolled) >= 8:
            sc = np.array([scores[(t, p)] for p in rolled])
            su = np.array([succ[(t, p)] for p in rolled])
            pick, oracle = su[np.argmax(sc)], su.max()
            m["top1_success"] = round(float(pick), 1)
            m["top1_regret"] = round(float(oracle - pick), 1)
            m["random_pick"] = round(float(su.mean()), 1)
            from scipy.stats import spearmanr
            m["spearman_own"] = round(float(spearmanr(sc, su)[0]), 3)
            pairs = [(scores[(t, a)] - scores[(t, b)], succ[(t, a)] - succ[(t, b)])
                     for a, b in itertools.combinations(rolled, 2)
                     if abs(succ[(t, a)] - succ[(t, b)]) > 10]
            if pairs:
                acc = np.mean([(ds > 0) == (dsu > 0) for ds, dsu in pairs])
                m["pairwise_acc"] = round(float(acc), 3)
                pooled_pairs += pairs
        if own and cross:
            y = np.array([1] * len(own) + [0] * len(cross))
            s = np.array([scores[(t, p)] for p in own + cross])
            m["relevance_auc"] = round(auc(y, s), 3)
        if rts and own:
            allp = own + rts
            s = np.array([scores[(t, p)] for p in allp])
            ranks = len(allp) + 1 - pd.Series(s).rank().values  # 1 = best score
            m["rt_mean_rank"] = round(float(np.mean([ranks[allp.index(p)] for p in rts])), 1)
            m["rt_n_phrases"] = len(allp)
        per_task[t] = m
    pooled = {}
    if pooled_pairs:
        pooled["pairwise_acc"] = round(float(np.mean([(a > 0) == (b > 0) for a, b in pooled_pairs])), 3)
        pooled["n_pairs"] = len(pooled_pairs)
    return {"per_task": per_task, "pooled": pooled}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--out", default="results/verifier_study_eval.json")
    args = ap.parse_args()

    phr = pd.read_parquet("results/phrase_artifacts/study_phrases_all.parquet")
    succ = load_success()
    feats = study_feature_rows()
    print(f"study feature rows: {len(feats)} | rolled (task,phrase) with success: {len(succ)}")

    report = {}
    for name, sc in baseline_scores(feats).items():
        report[name] = metrics_for(sc, phr, succ)
    for prefix in args.models:
        sc = model_scores(prefix, feats)
        if sc is not None:
            report[os.path.basename(prefix)] = metrics_for(sc, phr, succ)

    json.dump(report, open(args.out, "w"), indent=1)
    for name, r in report.items():
        pool = r["pooled"]
        print(f"\n== {name} == pooled pairwise_acc={pool.get('pairwise_acc')} (n={pool.get('n_pairs')})")
        for t, m in r["per_task"].items():
            print(f"  {t.replace('widowx_',''):28s} {m}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
