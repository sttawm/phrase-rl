"""Extract scoring contexts from RECORDED SIM trajectories (successful episodes).

User idea (2026-07-09): use successful rollouts' executed actions as ground truth —
contexts (sim frame, state, executed 4-step chunk) put the flow reward in the SAME
domain as rollout success, removing the real->sim confound from the bake-off.

  python -m phrase_rl.sim_contexts_extract --record-dir data/sim_traj \
    --out data/contexts_sim.parquet --per-episode 3
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

CHUNK = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-episode", type=int, default=3)
    ap.add_argument("--success-only", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows, eid = [], 0
    files = sorted(glob.glob(os.path.join(args.record_dir, "*.npz")))
    for f in files:
        z = np.load(f, allow_pickle=True)
        if args.success_only and not bool(z["success"]):
            continue
        imgs, states, actions = z["imgs"], z["states"], z["actions"]
        T = len(actions)
        if T < CHUNK + 2:
            continue
        n = min(args.per_episode, T - CHUNK)
        for t in sorted(rng.choice(T - CHUNK, size=n, replace=False).tolist()):
            rows.append({
                "task": str(z["task"]), "episode_index": eid, "t": int(t),
                "instruction": str(z["phrase"]),
                "action_chunk": actions[t:t + CHUNK].astype(np.float32).flatten(),
                "state": states[t].astype(np.float32),
                "image_png": bytes(imgs[t]),
            })
        eid += 1
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"{len(files)} episodes scanned -> {eid} successful used -> {len(df)} sim contexts")
    if len(df):
        print(df.groupby("task").size().to_string())


if __name__ == "__main__":
    main()
