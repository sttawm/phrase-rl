"""Re-score the study phrases with raw velocity capture (v, u per draw) so the
flow/both verifiers can be evaluated on the study's rollout-success labels.
Same seed/K as verifier training => identical CRN slots.

Pod (.venv), ~25 min:
  .venv/bin/python -m phrase_rl.study_verbose_rescore \
    --contexts data/contexts_0c_match.parquet \
    --phrases data/study_phrases_all.parquet \
    --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
    --k 8 --out data/study_verbose_flow.parquet
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
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    phrases = pd.read_parquet(args.phrases)
    tasks = [t for t in phrases["task"].unique() if t in set(contexts["task"])]

    PI0Policy = import_pi0_policy()
    policy = PI0Policy.from_pretrained(args.ckpt).to("cuda" if torch.cuda.is_available() else "cpu")
    scorer = Pi0PhraseScorer(policy, k=args.k, seed=args.seed, micro_batch=64)

    rows, done = [], set()
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        rows = prev.to_dict("records")
        done = set(zip(prev.task, prev.episode_index, prev.t))

    for task in tasks:
        items = list(dict.fromkeys(phrases[phrases.task == task]["phrase"]))
        for row in tqdm(list(contexts[contexts.task == task].itertuples()), desc=task):
            if (task, row.episode_index, row.t) in done:
                continue
            img = np.asarray(Image.open(io.BytesIO(row.image_png))).astype(np.float32) / 255.0
            losses, v, u = scorer.score_verbose(
                img.transpose(2, 0, 1), np.asarray(row.state),
                np.asarray(row.action_chunk), items)
            for pi, p in enumerate(items):
                for k in range(args.k):
                    rows.append({"task": task, "episode_index": row.episode_index, "t": row.t,
                                 "phrase": p, "k": k, "loss": float(losses[pi, k]),
                                 "v": v[pi, k].reshape(-1).astype(np.float32),
                                 "u": u[pi, k].reshape(-1).astype(np.float32)})
            pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
