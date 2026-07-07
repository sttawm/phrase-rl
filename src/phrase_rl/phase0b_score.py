"""Phase 0b: CRN-score every (context, phrase) pair with a frozen pi0 checkpoint.

For each context: [original] + each arm's rephrases (deduped, capped) are scored
under the SAME K (eps, tau) draws. Output is long format — one row per
(context, phrase, draw) — so analysis can do split-half reliability, per-draw
z-scoring, and variance decomposition without re-scoring.

Duplicate-phrase controls are intentionally absent: with CRN + deterministic
forward, identical text gives bit-identical losses (verified in smoke_pi0), so
the noise floor is measured by split-half disagreement instead.

Usage (on pod, .venv):
  python -m phrase_rl.phase0b_score \
    --contexts data/contexts_val_0b.parquet \
    --arm gemini=data/rephrases_val_0b.parquet \
    --arm qwen=data/qwen_rephrases_val_0b.parquet \
    --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
    --out data/scores_0b_rephrase.parquet
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


def dedup_keep_order(phrases, seen):
    out = []
    for p in phrases:
        key = str(p).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(str(p).strip())
    return out


def assemble_phrases(row, arm_frames, cap):
    """[(arm, phrase), ...] — original first, then each arm deduped."""
    seen = set()
    items = [("original", p) for p in dedup_keep_order([row["instruction"]], seen)]
    for arm, frame in arm_frames.items():
        match = frame[(frame["episode_index"] == row["episode_index"]) & (frame["t"] == row["t"])]
        if len(match) == 0:
            continue
        phrases = dedup_keep_order(list(match.iloc[0]["rephrases"])[:cap], seen)
        items.extend((arm, p) for p in phrases)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--arm", action="append", required=True, help="name=path.parquet")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=32, help="max rephrases per arm per context")
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()

    arm_frames = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        arm_frames[name] = pd.read_parquet(path)

    contexts = pd.read_parquet(args.contexts)
    done_frames, done_keys = [], set()
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        done_frames.append(prev)
        done_keys = set(zip(prev["episode_index"], prev["t"]))
        print(f"resume: {len(done_keys)} contexts already scored")

    PI0Policy = import_pi0_policy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PI0Policy.from_pretrained(args.ckpt).to(device)
    scorer = Pi0PhraseScorer(policy, k=args.k, seed=args.seed, micro_batch=args.micro_batch)
    tau = scorer.tau.numpy()

    pending = contexts[~contexts.apply(lambda r: (r["episode_index"], r["t"]) in done_keys, axis=1)]
    new_rows = 0
    for i, (_, row) in enumerate(tqdm(pending.iterrows(), total=len(pending), desc=f"score {args.ckpt.split('/')[-1]}")):
        items = assemble_phrases(row, arm_frames, args.cap)
        if len(items) < 8:
            continue
        img = np.asarray(Image.open(io.BytesIO(row["image_png"]))).astype(np.float32) / 255.0
        losses = scorer.score(
            img.transpose(2, 0, 1), np.asarray(row["state"]),
            np.asarray(row["action_chunk"]), [p for _, p in items],
        )  # (P, K)
        P, K = losses.shape
        done_frames.append(
            pd.DataFrame(
                {
                    "episode_index": np.repeat(row["episode_index"], P * K),
                    "t": np.repeat(row["t"], P * K),
                    "arm": np.repeat([a for a, _ in items], K),
                    "phrase": np.repeat([p for _, p in items], K),
                    "k": np.tile(np.arange(K), P),
                    "tau": np.tile(tau, P),
                    "loss": losses.flatten(),
                }
            )
        )
        new_rows += 1
        if (i + 1) % args.save_every == 0:
            pd.concat(done_frames, ignore_index=True).to_parquet(args.out, index=False)

    out = pd.concat(done_frames, ignore_index=True)
    out.to_parquet(args.out, index=False)
    print(f"scored {new_rows} new contexts -> {args.out} ({len(out)} rows total)")


if __name__ == "__main__":
    main()
