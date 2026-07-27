#!/usr/bin/env python3
"""Build v7e-strong SCORING context tables: for every full-Bridge instruction
with >= --min-eps episodes (empty label excluded), sample up to
--per-instruction episodes and extract --points frames each (image + a* chunk +
state), mirroring multi_t_contexts' per-episode hub pull. No traces needed —
these contexts are for pi0 reward scoring only (parents come from the traced
pool). Resumable: skips episodes already in --out if it exists.

  .venv-gen/bin/python scripts/build_club_contexts.py \
     --meta $META --out data/contexts_club.parquet
"""
import argparse
import collections
import io
import json
import os

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

import sys
sys.path.insert(0, "src")
from phrase_rl.extract_contexts import CHUNK_SIZE, CHUNKS_SIZE, IMAGE_KEY, REPO, decode_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="episodes.jsonl from the lerobot dataset cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-eps", type=int, default=16)
    ap.add_argument("--per-instruction", type=int, default=20)
    ap.add_argument("--points", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    metas = []
    for line in open(args.meta):
        e = json.loads(line)
        t = e.get("tasks")
        instr = t[0] if isinstance(t, list) and t else str(t)
        if instr.strip():
            metas.append({"episode_index": int(e["episode_index"]),
                          "episode_length": int(e["length"]), "instruction": instr})
    n = collections.Counter(m["instruction"] for m in metas)
    club = {i for i, c in n.items() if c >= args.min_eps}
    print(f"club: {len(club)} instructions (>= {args.min_eps} eps)")

    rng = np.random.default_rng(args.seed)
    by_instr = collections.defaultdict(list)
    for m in metas:
        if m["instruction"] in club:
            by_instr[m["instruction"]].append(m)
    picked = []
    for i, eps in sorted(by_instr.items()):
        idx = rng.choice(len(eps), size=min(args.per_instruction, len(eps)), replace=False)
        picked += [eps[j] for j in idx]
    print(f"picked {len(picked)} episodes")

    done = set()
    rows = []
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        done = set(prev.episode_index.unique())
        rows = prev.to_dict("records")
        print(f"resuming: {len(done)} episodes already extracted")

    flushed = len(rows)
    for k, m in enumerate(tqdm(picked, desc="episodes")):
        ep_idx, ep_len = m["episode_index"], m["episode_length"]
        if ep_idx in done or ep_len < CHUNK_SIZE + 2:
            continue
        try:
            dir_chunk = ep_idx // CHUNKS_SIZE
            pq = pd.read_parquet(hf_hub_download(
                REPO, f"data/chunk-{dir_chunk:03d}/episode_{ep_idx:06d}.parquet", repo_type="dataset"))
            vid = hf_hub_download(
                REPO, f"videos/chunk-{dir_chunk:03d}/{IMAGE_KEY}/episode_{ep_idx:06d}.mp4",
                repo_type="dataset")
            hi = ep_len - CHUNK_SIZE
            bounds = np.linspace(0, hi + 1, args.points + 1)
            ts = sorted({int(rng.integers(int(bounds[i]), max(int(bounds[i + 1]), int(bounds[i]) + 1)))
                         for i in range(args.points)})
            for t in ts:
                img = decode_frame(vid, t)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                rows.append({"instruction": m["instruction"], "episode_index": ep_idx,
                             "episode_length": ep_len, "t": t,
                             "action_chunk": np.stack(pq["action"].values[t:t + CHUNK_SIZE]).astype(np.float32).flatten(),
                             "state": np.asarray(pq["observation.state"].values[t], dtype=np.float32),
                             "image_png": buf.getvalue()})
        except Exception as e:
            print(f"[skip ep {ep_idx}] {type(e).__name__}: {e}", flush=True)
            continue
        if len(rows) - flushed >= 400:
            pd.DataFrame(rows).to_parquet(args.out, index=False)
            flushed = len(rows)
            print(f"[flush] {len(rows)} rows after {k+1}/{len(picked)} episodes", flush=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"DONE: {len(rows)} context rows -> {args.out}")


if __name__ == "__main__":
    main()
