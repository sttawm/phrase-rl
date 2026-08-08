#!/usr/bin/env python3
"""Score the whole phrase bank with the REAL reward channels, on a pod.

The bank was assembled from earlier experiments that recorded gripper error but
not the verifier channel, so 87% of its proxy values were imputed from a column
mean. The rules loop distils from that table, so it should rest on measurements
rather than fill-ins. This scores every non-sealed phrase properly:

    z    = -mean over ensemble members   (reward_mode="verifier")
    grip = the server's gripper sidecar

RAW channels are stored, never the proxy. The calibrated estimate is
    success = sigmoid(8.12 + 0.445*z + 11.32*(-grip))
(scripts/fit_simple_success_reward.py, fit against real rollout success rates),
and it is applied on READ. Storing z and grip means a recalibration re-reads a
night's scoring instead of invalidating it -- and the same F=4 aggregation the
calibration was fit on is used here, so the coefficients transfer directly.

Resumable and shard-able: pass --shard i --of n to split across pods; results
append to a per-shard parquet and are flushed after every task, so an interrupted
run resumes where it stopped rather than starting over.

  RUN on a pod, from /workspace/phrase-rl, with a verifier score server up:
    .venv-gen/bin/python scripts/score_bank.py --shard 0 --of 2
"""
import argparse
import os
import sys
import types
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/workspace/phrase-rl")
sys.path.insert(0, str(REPO / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--of", type=int, default=1)
ap.add_argument("--contexts", type=int, default=16)   # C
ap.add_argument("--frames", type=int, default=4)      # F, the calibration's value
ap.add_argument("--budget", type=int, default=64)     # F*C target
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--ipc", default=os.environ.get("IPC_DIR", "/workspace/ipc_rules"))
args = ap.parse_args()

from phrase_rl.phase2_train import score_phrases  # noqa: E402

todo = pd.read_parquet(REPO / "results/analysis/bank_to_score.parquet")
tasks = sorted(todo.task.unique())
mine = [t for i, t in enumerate(tasks) if i % args.of == args.shard]
todo = todo[todo.task.isin(mine)]

out_path = REPO / f"results/analysis/bank_scores_shard{args.shard}of{args.of}.parquet"
done = set()
rows = []
if out_path.exists():
    prev = pd.read_parquet(out_path)
    rows = prev.to_dict("records")
    done = set(zip(prev.task, prev.phrase))
    print(f"resume: {len(done)} phrases already scored", flush=True)

banks = pd.concat([pd.read_parquet(f) for f in [
    REPO / "data/contexts_train_multit16.parquet",
    REPO / "data/contexts_val_multit16.parquet",
    REPO / "data/contexts_club.parquet"] if f.exists()], ignore_index=True)
bkey = "task" if "task" in banks.columns else "instruction"
banks["_key"] = banks[bkey].astype(str)

rng = np.random.default_rng(args.seed)
sargs = types.SimpleNamespace(k=8, score_seed=args.seed, tau_min=0.0,
                              reward_mode="verifier", k_l2=4, score_timeout=3600,
                              _reward_frames_map=None)
ipc = Path(args.ipc)
print(f"shard {args.shard}/{args.of}: {len(mine)} tasks, {len(todo)} phrases "
      f"({len(todo) - len(done)} outstanding)", flush=True)

for ti, (task, grp) in enumerate(todo.groupby("task"), 1):
    phrases = [p for p in dict.fromkeys(grp.phrase.astype(str))
               if (task, p) not in done]
    if not phrases:
        continue
    sub = banks[banks._key == str(task)]
    if len(sub) == 0:
        for p in phrases:
            rows.append({"task": task, "phrase": p, "z": np.nan, "grip": np.nan,
                         "n_ctx": 0})
        print(f"[{ti}/{len(mine)}] {str(task)[:44]!r}: NO CONTEXTS", flush=True)
        continue

    eps = sorted(sub.episode_index.unique())
    C = min(args.contexts, len(eps))
    F = int(np.clip(round(args.budget / max(C, 1)), args.frames, 16))
    pick = rng.choice(eps, size=C, replace=False)
    ctx_rows = []
    for e in sorted(pick):
        ep = sub[sub.episode_index == e]
        ep = ep.sort_values("t") if "t" in ep.columns else ep
        ctx_rows.extend(r for _, r in ep.head(F).iterrows())

    zs, gs = [], []
    for i in range(0, len(ctx_rows), 8):
        batch = ctx_rows[i:i + 8]
        losses = score_phrases(ipc, f"bank_{uuid.uuid4().hex[:8]}",
                               [(fr, phrases) for fr in batch], sargs)
        grips = getattr(score_phrases, "last_grips", [None] * len(losses))
        for L, G in zip(losses, grips):
            zs.append(-np.asarray(L, dtype=np.float64).mean(axis=1))
            gs.append(np.asarray(G, dtype=np.float64) if G is not None
                      else np.full(len(phrases), np.nan))
    zm, gm = np.nanmean(np.stack(zs), axis=0), np.nanmean(np.stack(gs), axis=0)
    for p, z, g in zip(phrases, zm, gm):
        rows.append({"task": task, "phrase": p, "z": float(z), "grip": float(g),
                     "n_ctx": len(ctx_rows)})
    pd.DataFrame(rows).to_parquet(out_path, index=False)   # flush per task
    print(f"[{ti}/{len(mine)}] {str(task)[:44]!r}: {len(phrases)} phrases "
          f"F={F} C={C} -> {len(rows)} rows total", flush=True)

pd.DataFrame(rows).to_parquet(out_path, index=False)
print(f"BANK-SCORING-DONE shard {args.shard}: {len(rows)} rows -> {out_path.name}")
