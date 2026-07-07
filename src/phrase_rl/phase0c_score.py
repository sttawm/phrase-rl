"""Score each SIMPLER task's phrases on its matched real Bridge contexts (CRN).

Per task: every phrase (original + rephrases) scored on each matched context under
the SAME K draws (CRN, per context). Long output: one row per
(task, context, phrase, draw). Analysis averages over context+draw -> one flow
loss per (task, phrase), to correlate against rollout success.

Run on pod, .venv (after rollouts free the GPU):
  python -m phrase_rl.phase0c_score \
    --contexts data/contexts_0c_match.parquet --phrases data/phrases_0c.parquet \
    --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --out data/scores_0c.parquet
"""

import argparse
import io
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from phrase_rl.pi0_scoring import Pi0PhraseScorer, import_pi0_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--phrases", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro-batch", type=int, default=64)
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    phrases = pd.read_parquet(args.phrases)
    tasks = [t for t in phrases["task"].unique() if t in set(contexts["task"])]
    print(f"scoring tasks with matched contexts: {tasks}")

    PI0Policy = import_pi0_policy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PI0Policy.from_pretrained(args.ckpt).to(device)
    scorer = Pi0PhraseScorer(policy, k=args.k, seed=args.seed, micro_batch=args.micro_batch)
    tau = scorer.tau.numpy()

    frames = []
    done_keys = set()
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out); frames.append(prev)
        done_keys = set(zip(prev.task, prev.episode_index, prev.t))

    for task in tasks:
        tphr = phrases[phrases.task == task]
        items = list(dict.fromkeys(tphr["phrase"]))  # dedupe, keep order
        arm_of = dict(zip(tphr["phrase"], tphr["arm"]))
        tctx = contexts[contexts.task == task]
        for row in tqdm(list(tctx.itertuples()), desc=task):
            if (task, row.episode_index, row.t) in done_keys:
                continue
            img = np.asarray(Image.open(io.BytesIO(row.image_png))).astype(np.float32) / 255.0
            losses = scorer.score(img.transpose(2, 0, 1), np.asarray(row.state),
                                  np.asarray(row.action_chunk), items)  # (P, K)
            P, K = losses.shape
            frames.append(pd.DataFrame({
                "task": task, "episode_index": row.episode_index, "t": row.t,
                "arm": np.repeat([arm_of[p] for p in items], K),
                "phrase": np.repeat(items, K),
                "k": np.tile(np.arange(K), P), "tau": np.tile(tau, P),
                "loss": losses.flatten(),
            }))
            pd.concat(frames, ignore_index=True).to_parquet(args.out, index=False)

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
