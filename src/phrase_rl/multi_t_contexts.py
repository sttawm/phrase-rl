"""Expand an existing contexts parquet to multiple timesteps per episode.

Given contexts (one frame/episode), re-pull each episode from the Bridge hub
(selective per-episode files) and emit ONE context per quartile of the valid
t-range (4 rows/episode, seeded uniform within each quartile). All non-frame
columns (task, instruction, ...) are carried over from the source row.

Eval-only companion to the verifier: the model trained on uniform-t single
frames, so quartile frames are in-distribution; this just multiplies evidence.

  python -m phrase_rl.multi_t_contexts --contexts data/contexts_0c_match.parquet \
    --out data/contexts_0c_match_multit.parquet
"""

import argparse
import io

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from phrase_rl.extract_contexts import (CHUNK_SIZE, CHUNKS_SIZE, IMAGE_KEY, REPO,
                                        decode_frame, load_episode_meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", type=int, default=4, help="one per equal segment of valid t-range")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = pd.read_parquet(args.contexts)
    if "episode_length" not in src.columns:
        lengths = {e["episode_index"]: e["length"] for e in load_episode_meta()}
        src = src.assign(episode_length=src.episode_index.map(lengths))
        assert src.episode_length.notna().all(), "episode(s) missing from hub meta"
    rng = np.random.default_rng(args.seed)
    carry = [c for c in src.columns if c not in ("t", "action_chunk", "state", "image_png")]

    rows = []
    for r in tqdm(list(src.itertuples()), desc="episodes"):
        ep_idx, ep_len = int(r.episode_index), int(r.episode_length)
        hi = ep_len - CHUNK_SIZE  # inclusive max t
        dir_chunk = ep_idx // CHUNKS_SIZE
        pq = pd.read_parquet(hf_hub_download(
            REPO, f"data/chunk-{dir_chunk:03d}/episode_{ep_idx:06d}.parquet", repo_type="dataset"))
        vid = hf_hub_download(
            REPO, f"videos/chunk-{dir_chunk:03d}/{IMAGE_KEY}/episode_{ep_idx:06d}.mp4",
            repo_type="dataset")
        bounds = np.linspace(0, hi + 1, args.points + 1)
        ts = sorted({int(rng.integers(int(bounds[i]), max(int(bounds[i + 1]), int(bounds[i]) + 1)))
                     for i in range(args.points)})
        for t in ts:
            img = decode_frame(vid, t)
            buf = io.BytesIO(); img.save(buf, format="PNG")
            row = {c: getattr(r, c) for c in carry}
            row.update({
                "t": t,
                "action_chunk": np.stack(pq["action"].values[t:t + CHUNK_SIZE]).astype(np.float32).flatten(),
                "state": np.asarray(pq["observation.state"].values[t], dtype=np.float32),
                "image_png": buf.getvalue(),
            })
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    print(f"wrote {len(out)} context-points from {len(src)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
