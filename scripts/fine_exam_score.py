#!/usr/bin/env python3
"""Fine-discrimination exam scorer: score every panel phrase on every context
frame of its task via the score server (reward_mode=verifier). Each (episode, t)
is submitted as its OWN context so the F axis stays available downstream.

Outputs one row per (task, phrase, episode_index, t):
  z_row  = -mean(losses over k)   (ensemble-calibrated logit reward)
  grip_row = server grip error
Run on a pod with the score server up on --ipc-dir:
  PYTHONPATH=src .venv-gen/bin/python scripts/fine_exam_score.py \
    --contexts data/contexts_sim_oov.parquet \
    --phrases results/analysis/fine_exam_phrases.parquet \
    --out results/analysis/fine_exam_features_oov.parquet --ipc-dir /workspace/ipc6
"""
import argparse
import os
import uuid
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from phrase_rl.phase2_train import score_phrases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--phrases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ipc-dir", required=True)
    ap.add_argument("--chunk", type=int, default=36, help="frame-contexts per server job")
    args = ap.parse_args()

    sargs = Namespace(k=8, score_seed=0, tau_min=0.0, reward_mode="verifier",
                      k_l2=4, score_timeout=3600, _reward_frames_map=None)
    ctx = pd.read_parquet(args.contexts)
    pan = pd.read_parquet(args.phrases)
    ipc = Path(args.ipc_dir)

    done = set()
    rows = []
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        rows = prev.to_dict("records")
        done = set(zip(prev.task, prev.episode_index, prev.t))
        print(f"resume: {len(done)} (task,ep,t) cells already scored")

    for task, cgrp in ctx.groupby("task"):
        phrases = list(pan[pan.task == task].phrase)
        if not phrases:
            print(f"[skip] no panel phrases for {task}")
            continue
        frames = [r for r in cgrp.to_dict("records")
                  if (task, r["episode_index"], r["t"]) not in done]
        print(f"{task}: {len(frames)} frame-contexts x {len(phrases)} phrases")
        for i in range(0, len(frames), args.chunk):
            batch = frames[i:i + args.chunk]
            contexts = [(fr, phrases) for fr in batch]
            job = f"fexam_{uuid.uuid4().hex[:8]}"
            losses = score_phrases(ipc, job, contexts, sargs)
            grips = getattr(score_phrases, "last_grips", [None] * len(losses))
            for fr, L, G in zip(batch, losses, grips):
                z = -np.asarray(L).mean(axis=1)
                for pi, p in enumerate(phrases):
                    rows.append({"task": task, "phrase": p,
                                 "episode_index": int(fr["episode_index"]), "t": int(fr["t"]),
                                 "z_row": float(z[pi]),
                                 "grip_row": float(G[pi]) if G is not None else np.nan})
            pd.DataFrame(rows).to_parquet(args.out, index=False)
            print(f"  [{i + len(batch)}/{len(frames)}] flushed {len(rows)} rows", flush=True)
    print(f"DONE: {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
