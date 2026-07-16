"""Ground-truth action-history features for verifier rows — dataset reads only.

For every (episode_index, t) in the given feature/context parquets, fetch the
episode's action table from the Bridge hub (tiny per-episode parquet) and emit:
  history: last HIST_STEPS raw actions before t, zero-padded at episode start
           ((HIST_STEPS*7,) float32)
  t_frac:  t / (episode_length - CHUNK_SIZE)  — episode-progress scalar

History is the demo's executed actions (the "correct phrase" trajectory) — the
same for all conditions of a context, so it cannot shortcut match/mismatch; it
only helps interpret pi0's response at mid-episode frames.

  python -m phrase_rl.action_history \
    --inputs results/overnight/raw/verifier_features_train.parquet \
             results/overnight/raw/verifier_features_val.parquet \
             data/contexts_0c_match.parquet \
    --out data/action_history.parquet
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from phrase_rl.extract_contexts import CHUNK_SIZE, CHUNKS_SIZE, REPO, load_episode_meta

HIST_STEPS = 16  # 4 chunks of raw 7-DoF actions


def episode_actions(ep_idx: int) -> np.ndarray:
    p = hf_hub_download(REPO, f"data/chunk-{ep_idx // CHUNKS_SIZE:03d}/episode_{ep_idx:06d}.parquet",
                        repo_type="dataset")
    return np.stack(pd.read_parquet(p, columns=["action"])["action"].values).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    keys = set()
    for f in args.inputs:
        d = pd.read_parquet(f, columns=["episode_index", "t"])
        keys |= set(zip(d.episode_index.astype(int), d.t.astype(int)))
    eps = sorted({e for e, _ in keys})
    print(f"{len(keys)} (episode, t) keys over {len(eps)} episodes")

    lengths = {e["episode_index"]: e["length"] for e in load_episode_meta()}
    acts = {}
    with ThreadPoolExecutor(args.workers) as ex:
        for e, a in zip(eps, tqdm(ex.map(episode_actions, eps), total=len(eps), desc="episodes")):
            acts[e] = a

    rows = []
    for e, t in sorted(keys):
        a = acts[e]
        lo = max(0, t - HIST_STEPS)
        h = a[lo:t]
        if len(h) < HIST_STEPS:
            h = np.concatenate([np.zeros((HIST_STEPS - len(h), a.shape[1]), np.float32), h])
        rows.append({"episode_index": e, "t": t,
                     "history": h.flatten(),
                     "t_frac": float(t / max(1, lengths[e] - CHUNK_SIZE))})
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} history rows -> {args.out}")


if __name__ == "__main__":
    main()
