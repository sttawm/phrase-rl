"""Verifier training data: pi0 behavioral features under matched vs mismatched instructions.

For each Bridge context (o, state, a*), score THREE conditions with CRN-matched
draws (same fixed (eps, tau) K draws and decode noises for all conditions and all
contexts — tau is constant per feature slot by construction):
  own  (label 1): the episode's own instruction — the one actually paired with a*
  hard (label 0): another episode's instruction sharing >=1 content token
  easy (label 0): another episode's instruction sharing no content tokens

No rephrases, no phrase text in features: the classifier sees only pi0's response
(raw velocity vectors v_theta and targets u per draw, decoded action chunks) plus
ground truth. One row per (context, condition); wide list columns.

Run on pod (.venv):
  .venv/bin/python -m phrase_rl.build_verifier_data \
    --contexts data/contexts_train.parquet --stats-contexts data/contexts_train.parquet \
    --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
    --k 8 --k-decode 4 --out data/verifier_features_train.parquet
"""

import argparse
import io
import os
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from phrase_rl.pi0_scoring import Pi0PhraseScorer, import_pi0_policy

STOP = {"put", "the", "place", "move", "pick", "up", "on", "in", "to", "of", "a", "an",
        "and", "it", "its", "from", "with", "onto", "into", "right", "left", "front",
        "behind", "top", "side", "then", "please", "next", "down", "over", "at", "is"}


def content_tokens(s: str) -> set:
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2}


def sample_negatives(df: pd.DataFrame, seed: int):
    """Per row: (hard_idx, easy_idx) into df. Hard = max token overlap (random among
    top-5, overlap >= 1); easy = zero overlap. Deterministic in seed."""
    rng = np.random.default_rng(seed)
    toks = [content_tokens(s) for s in df.instruction]
    texts = [s.strip().lower() for s in df.instruction]
    hard_idx, easy_idx = [], []
    n = len(df)
    for i in range(n):
        ov = np.array([len(toks[i] & toks[j]) if (j != i and texts[j] != texts[i]) else -1
                       for j in range(n)])
        cand = np.where(ov >= 1)[0]
        if len(cand):
            top = cand[np.argsort(-ov[cand])][:5]
            hard_idx.append(int(rng.choice(top)))
        else:  # no overlapping episode: fall back to any other (still a valid negative)
            hard_idx.append(int(rng.choice(np.where(ov == 0)[0])))
        zero = np.where(ov == 0)[0]
        assert len(zero), f"context {i} overlaps every other instruction?!"
        easy_idx.append(int(rng.choice(zero)))
    return hard_idx, easy_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--stats-contexts", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k-decode", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap contexts (0 = all)")
    args = ap.parse_args()

    df = pd.read_parquet(args.contexts)
    if args.limit:
        df = df.iloc[: args.limit].reset_index(drop=True)
    stats = pd.read_parquet(args.stats_contexts, columns=["action_chunk"])
    chunks = np.stack(stats["action_chunk"].map(np.asarray))
    dim_std = chunks.reshape(-1, 4, 7).std(axis=(0, 1)) + 1e-8

    hard_idx, easy_idx = sample_negatives(df, args.seed)
    n_hard_fallback = sum(1 for i, h in enumerate(hard_idx)
                          if not (content_tokens(df.instruction[i]) & content_tokens(df.instruction[h])))
    print(f"{len(df)} contexts | hard negatives without token overlap (fallback): {n_hard_fallback}")

    PI0Policy = import_pi0_policy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PI0Policy.from_pretrained(args.ckpt).to(device)
    scorer = Pi0PhraseScorer(policy, k=args.k, seed=args.seed, micro_batch=64)

    rows = []
    done_keys = set()
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        rows = prev.to_dict("records")
        done_keys = set(zip(prev.episode_index, prev.t))  # (ep, t): multi-t contexts share episodes
        print(f"resume: {len(done_keys)} context-points already featurized")

    for i, row in enumerate(tqdm(list(df.itertuples()), desc="contexts")):
        if (row.episode_index, row.t) in done_keys:
            continue
        img = np.asarray(Image.open(io.BytesIO(row.image_png))).astype(np.float32) / 255.0
        img_chw = img.transpose(2, 0, 1)
        state = np.asarray(row.state)
        a_star = np.asarray(row.action_chunk, dtype=np.float32)
        conds = [("own", 1, str(row.instruction)),
                 ("hard", 0, str(df.instruction[hard_idx[i]])),
                 ("easy", 0, str(df.instruction[easy_idx[i]]))]
        phrases = [c[2] for c in conds]
        losses, v, u = scorer.score_verbose(img_chw, state, a_star, phrases)
        dec, nl2, grip = scorer.decode_verbose(img_chw, state, a_star, phrases,
                                               dim_std, k_l2=args.k_decode)
        for j, (role, label, text) in enumerate(conds):
            rows.append({
                "episode_index": int(row.episode_index), "t": int(row.t),
                "role": role, "label": label, "instruction": text,
                "flow_loss": losses[j].astype(np.float32),          # (K,)
                "flow_v": v[j].reshape(-1).astype(np.float32),      # (K*H*7,)
                "flow_u": u[j].reshape(-1).astype(np.float32),      # (K*H*7,)
                "decoded": dec[j].reshape(-1).astype(np.float32),   # (Kd*H*7,)
                "norm_l2": nl2[j].astype(np.float32),               # (Kd,)
                "grip_err": grip[j].astype(np.float32),             # (Kd,)
                "a_star": a_star.reshape(-1).astype(np.float32),    # (H*7,)
                "state": state.astype(np.float32),
            })
        if (i + 1) % 25 == 0:
            pd.DataFrame(rows).to_parquet(args.out, index=False)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    meta = {"k": args.k, "k_decode": args.k_decode, "seed": args.seed,
            "tau": scorer.tau.tolist(), "dim_std": dim_std.tolist(), "ckpt": args.ckpt}
    import json
    json.dump(meta, open(os.path.splitext(args.out)[0] + "_meta.json", "w"), indent=1)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
