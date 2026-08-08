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
ap.add_argument("--task-kind", default="all", choices=["all", "train", "sim"],
                help="train = the bridge training instructions; sim = the "
                     "simulator tasks (widowx_*), which the loop validates on")
ap.add_argument("--contexts", type=int, default=16)   # C
ap.add_argument("--frames", type=int, default=4)      # F, the calibration's value
ap.add_argument("--budget", type=int, default=64)     # F*C target
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--phrase-chunk", type=int, default=8,
                help="phrases carried to completion together before flushing")
ap.add_argument("--call-rows", type=int, default=64,
                help="frames sent per IPC call for the phrase in hand")
ap.add_argument("--ipc", default=os.environ.get("IPC_DIR", "/workspace/ipc_rules"))
args = ap.parse_args()

from phrase_rl.phase2_train import score_phrases  # noqa: E402

todo = pd.read_parquet(REPO / "results/analysis/bank_to_score.parquet")
is_sim = todo.task.astype(str).str.startswith("widowx_")
if args.task_kind == "train":
    todo = todo[~is_sim]
elif args.task_kind == "sim":
    todo = todo[is_sim]
tasks = sorted(todo.task.unique())
mine = [t for i, t in enumerate(tasks) if i % args.of == args.shard]
todo = todo[todo.task.isin(mine)]

out_path = REPO / (f"results/analysis/bank_scores_{args.task_kind}"
                   f"_{args.shard}of{args.of}.parquet")
done = set()
rows = []
if out_path.exists():
    prev = pd.read_parquet(out_path)
    rows = prev.to_dict("records")
    done = set(zip(prev.task, prev.phrase))
    print(f"resume: {len(done)} phrases already scored", flush=True)

# Context tables disagree on their key column: the club/val tables are keyed by
# instruction text, the sim tables by task name. Build _key PER FILE before
# concatenating -- deciding once on the concatenated frame picks "task", which is
# null on every club row, and silently yields NO CONTEXTS for all 213 training
# tasks.
parts = []
for f in [REPO / "data/contexts_train_multit16.parquet",
          REPO / "data/contexts_val_multit16.parquet",
          REPO / "data/contexts_club.parquet",
          REPO / "data/contexts_sim_oov.parquet",
          REPO / "data/contexts_sim_oov5.parquet",
          REPO / "data/contexts_sim_val8.parquet"]:
    if not f.exists():
        continue
    d = pd.read_parquet(f)
    k = "task" if "task" in d.columns else "instruction"
    d["_key"] = d[k].astype(str)
    parts.append(d)
    print(f"contexts: {f.name} ({len(d)} rows, keyed on {k})", flush=True)
if not parts:
    sys.exit("no context tables found on this pod")
banks = pd.concat(parts, ignore_index=True)

# The bank names sim tasks widowx_<x>; some context tables use the bare <x>.
alias = {}
for k in banks._key.unique():
    alias.setdefault(f"widowx_{k}", k)
    if str(k).startswith("widowx_"):
        alias.setdefault(str(k)[len("widowx_"):], k)


def contexts_for(task):
    t = str(task)
    sub = banks[banks._key == t]
    if len(sub) == 0 and t in alias:
        sub = banks[banks._key == alias[t]]
    if len(sub) == 0 and t.endswith("_clean"):
        sub = banks[banks._key == t[: -len("_clean")]]
    return sub

rng = np.random.default_rng(args.seed)
sargs = types.SimpleNamespace(k=8, score_seed=args.seed, tau_min=0.0,
                              reward_mode="verifier", k_l2=4, score_timeout=3600,
                              _reward_frames_map=None)
ipc = Path(args.ipc)
print(f"[{args.task_kind}] shard {args.shard}/{args.of}: {len(mine)} tasks, {len(todo)} phrases "
      f"({len(todo) - len(done)} outstanding)", flush=True)

for ti, (task, grp) in enumerate(todo.groupby("task"), 1):
    phrases = [p for p in dict.fromkeys(grp.phrase.astype(str))
               if (task, p) not in done]
    if not phrases:
        continue
    sub = contexts_for(task)
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

    # DEPTH-FIRST IN CHUNKS. A phrase is carried to completion over all F*C
    # frames and flushed before the next chunk starts, so an interruption costs
    # at most one chunk rather than a whole task.
    #
    # Chunk rather than one-at-a-time because the server batches over the PHRASE
    # LIST within a context: it bills ~2.7s per context regardless of how many
    # phrases ride along (GPU sits at ~10% with a single phrase). One phrase per
    # call therefore costs a factor of P, not nothing -- 118h instead of 15h for
    # this bank. A chunk of 8 recovers essentially all of the batching.
    for ci in range(0, len(phrases), args.phrase_chunk):
        chunk = phrases[ci:ci + args.phrase_chunk]
        zs, gs = [], []
        for i in range(0, len(ctx_rows), args.call_rows):
            batch = ctx_rows[i:i + args.call_rows]
            losses = score_phrases(ipc, f"bank_{uuid.uuid4().hex[:8]}",
                                   [(fr, chunk) for fr in batch], sargs)
            grips = getattr(score_phrases, "last_grips", [None] * len(losses))
            for L, G in zip(losses, grips):
                zs.append(-np.asarray(L, dtype=np.float64).mean(axis=1))
                gs.append(np.asarray(G, dtype=np.float64).ravel() if G is not None
                          else np.full(len(chunk), np.nan))
        zm = np.nanmean(np.stack(zs), axis=0)
        gm = np.nanmean(np.stack(gs), axis=0)
        for phr, z, g in zip(chunk, np.atleast_1d(zm), np.atleast_1d(gm)):
            rows.append({"task": task, "phrase": phr, "z": float(z),
                         "grip": float(g), "n_ctx": len(ctx_rows)})
        pd.DataFrame(rows).to_parquet(out_path, index=False)   # flush per CHUNK
        pi = min(ci + args.phrase_chunk, len(phrases))
        if pi % 5 == 0 or pi == len(phrases):
            print(f"[{ti}/{len(mine)}] {str(task)[:40]!r}: {pi}/{len(phrases)} "
                  f"phrases F={F} C={C} -> {len(rows)} rows total", flush=True)

pd.DataFrame(rows).to_parquet(out_path, index=False)
print(f"BANK-SCORING-DONE [{args.task_kind}] shard {args.shard}: "
      f"{len(rows)} rows -> {out_path.name}")
