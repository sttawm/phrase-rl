"""Train the verifier MLP: P(instruction matches context | pi0's behavioral response).

Bridge-only training (own=1 vs hard/easy=0), CE with pos_weight for the 1:2
imbalance. Val = episode-disjoint Bridge contexts. Reports AUC overall AND on
hard negatives only — the hard-negative AUC is the number that predicts
usefulness as a reward (easy relevance saturates first).

Local CPU, seconds-to-minutes:
  .venv/bin/python -m phrase_rl.train_verifier \
    --train results/overnight/raw/verifier_features_train.parquet \
    --val results/overnight/raw/verifier_features_val.parquet \
    --mode both --out results/checkpoints/verifier_both
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from phrase_rl.verifier_features import featurize


def auc(y, s):
    """Rank-based ROC AUC (no sklearn dep)."""
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class MLP(nn.Module):
    def __init__(self, d_in, hidden=(256, 64), p=0.15):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--mode", choices=["flow", "l2", "both"], required=True)
    ap.add_argument("--out", required=True, help="output prefix (writes .pt + .json)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-diff", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tr = pd.read_parquet(args.train)
    va = pd.read_parquet(args.val)
    Xtr, ytr = featurize(tr, args.mode, not args.no_diff), tr.label.values.astype(np.float32)
    Xva, yva = featurize(va, args.mode, not args.no_diff), va.label.values.astype(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    print(f"mode={args.mode} d={Xtr.shape[1]} | train {len(tr)} (pos {int(ytr.sum())}) "
          f"| val {len(va)} (pos {int(yva.sum())})")

    model = MLP(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / ytr.sum()))
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xv = torch.from_numpy(Xva)

    hard_mask = (va.role == "hard").values | (va.label == 1).values
    easy_mask = (va.role == "easy").values | (va.label == 1).values
    best_auc, best_state, best_ep, bad = -1, None, -1, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = perm[i : i + 256]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            sv = model(Xv).numpy()
        a = auc(yva, sv)
        if a > best_auc:
            best_auc, best_state, best_ep, bad = a, {k: v.clone() for k, v in model.state_dict().items()}, ep, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
        if ep % 25 == 0:
            print(f"  ep {ep}: val AUC {a:.4f} (best {best_auc:.4f}@{best_ep})")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        sv = model(Xv).numpy()
    metrics = {
        "mode": args.mode, "d_in": int(Xtr.shape[1]), "best_epoch": best_ep,
        "val_auc": auc(yva, sv),
        "val_auc_hard": auc(yva[hard_mask], sv[hard_mask]),
        "val_auc_easy": auc(yva[easy_mask], sv[easy_mask]),
        "n_train": len(tr), "n_val": len(va), "seed": args.seed,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "mode": args.mode, "d_in": Xtr.shape[1], "include_diff": not args.no_diff}, args.out + ".pt")
    json.dump(metrics, open(args.out + ".json", "w"), indent=1)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
