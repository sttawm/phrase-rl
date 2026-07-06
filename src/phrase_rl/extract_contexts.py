"""Extract scoring contexts (o_t, l, a*_t) from IPEC-COMMUNITY/bridge_orig_lerobot.

A context is one frame of one episode: the image_0 camera frame at timestep t,
the episode's instruction, the robot state at t, and the ground-truth action
chunk a*_{t..t+3} (chunk size 4, matching the INTACT pi0 checkpoints).

Episodes are assigned to train/val/test splits deterministically by hashing
episode_index (no split file needed; stable across machines). t is sampled
uniformly per episode (see EXPERIMENT.md, "Timestep choice").

Only the selected episodes' parquet + image_0 video files are downloaded.

Usage:
  python -m phrase_rl.extract_contexts --split val --n 250 --out data/contexts_val_0b.parquet
"""

import argparse
import io
import json

import av
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"
CHUNK_SIZE = 4  # pi0 action chunk length (INTACT checkpoints)
CHUNKS_SIZE = 1000  # episodes per directory chunk in the dataset layout
IMAGE_KEY = "observation.images.image_0"
# Knuth multiplicative hash → stable pseudo-uniform in [0,1) per episode.
SPLIT_FRACS = {"val": (0.00, 0.05), "test": (0.05, 0.10), "train": (0.10, 1.00)}


def episode_split_u(episode_index: int) -> float:
    return ((episode_index * 2654435761) % (2**32)) / 2**32


def load_episode_meta():
    p = hf_hub_download(REPO, "meta/episodes.jsonl", repo_type="dataset")
    with open(p) as f:
        return [json.loads(line) for line in f]


def decode_frame(video_path: str, frame_idx: int) -> Image.Image:
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_idx:
                return frame.to_image()
    raise ValueError(f"frame {frame_idx} not found in {video_path}")


def extract(split: str, n_contexts: int, seed: int, min_len: int) -> pd.DataFrame:
    lo, hi = SPLIT_FRACS[split]
    episodes = [
        e
        for e in load_episode_meta()
        if lo <= episode_split_u(e["episode_index"]) < hi
        and e["length"] >= min_len
        and e.get("tasks")
        and e["tasks"][0].strip()  # Bridge has episodes with empty instructions
    ]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(episodes), size=n_contexts, replace=False)

    rows = []
    for idx in tqdm(chosen, desc=f"extracting {split}"):
        ep = episodes[idx]
        ep_idx, ep_len = ep["episode_index"], ep["length"]
        dir_chunk = ep_idx // CHUNKS_SIZE
        # last valid t leaves a full action chunk: t + CHUNK_SIZE <= ep_len
        t = int(rng.integers(0, ep_len - CHUNK_SIZE + 1))

        pq_path = hf_hub_download(
            REPO, f"data/chunk-{dir_chunk:03d}/episode_{ep_idx:06d}.parquet", repo_type="dataset"
        )
        df = pd.read_parquet(pq_path)
        actions = np.stack(df["action"].values[t : t + CHUNK_SIZE])  # (4, 7)
        state = np.asarray(df["observation.state"].values[t])  # (8,)

        vid_path = hf_hub_download(
            REPO,
            f"videos/chunk-{dir_chunk:03d}/{IMAGE_KEY}/episode_{ep_idx:06d}.mp4",
            repo_type="dataset",
        )
        img = decode_frame(vid_path, t)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        rows.append(
            {
                "episode_index": ep_idx,
                "episode_length": ep_len,
                "t": t,
                "instruction": ep["tasks"][0],
                "action_chunk": actions.astype(np.float32).flatten(),  # (28,)
                "state": state.astype(np.float32),
                "image_png": buf.getvalue(),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLIT_FRACS), required=True)
    ap.add_argument("--n", type=int, required=True, help="number of contexts (1 per episode)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=CHUNK_SIZE + 4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = extract(args.split, args.n, args.seed, args.min_len)
    df.to_parquet(args.out, index=False)
    sizes = df["image_png"].map(len)
    print(
        f"wrote {len(df)} contexts to {args.out} "
        f"({sizes.sum() / 1e6:.1f} MB images, {df['instruction'].nunique()} unique instructions)"
    )


if __name__ == "__main__":
    main()
