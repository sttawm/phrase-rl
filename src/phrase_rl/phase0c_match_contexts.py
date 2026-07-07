"""Find real Bridge contexts matching each SIMPLER task, for 0c flow-loss scoring.

Flow loss needs a real (image, ground-truth action) whose action the task's
phrases plausibly describe. Bridge coverage is uneven (see MATCHERS): carrot is
exact and abundant; stack/spoon are semantic matches; eggplant-in-basket has no
Bridge counterpart and is excluded (keeps its rollout data, sits out the
loss↔success correlation). Split is ignored — these contexts only measure the
frozen reward's phrase ranking, never train the phrase model, so no leakage.

  python -m phrase_rl.phase0c_match_contexts --per-task 20 --out data/contexts_0c_match.parquet
"""

import argparse
import io

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from phrase_rl.extract_contexts import (
    CHUNK_SIZE, CHUNKS_SIZE, IMAGE_KEY, REPO, decode_frame, load_episode_meta,
)

# task -> predicate over the lowercased Bridge instruction. Matched action must be
# what the task's phrases describe (same object AND same target).
MATCHERS = {
    "widowx_carrot_on_plate": lambda s: s == "put carrot on plate",
    "widowx_stack_cube": lambda s: ("block" in s or "cube" in s) and "on top of" in s
        and ("yellow" in s or "green" in s),
    "widowx_spoon_on_towel": lambda s: "spoon" in s and "towel" in s
        and ("onto towel" in s or "on the towel" in s or "on towel" in s),
    # widowx_put_eggplant_in_basket: intentionally absent — no eggplant+basket in Bridge.
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    episodes = load_episode_meta()
    rng = np.random.default_rng(args.seed)
    rows = []
    for task, pred in MATCHERS.items():
        matched = [e for e in episodes
                   if (e.get("tasks") and pred(e["tasks"][0].strip().lower())
                       and e["length"] >= CHUNK_SIZE + 4)]
        if not matched:
            print(f"{task}: NO MATCHES"); continue
        pick = rng.choice(len(matched), size=min(args.per_task, len(matched)), replace=False)
        print(f"{task}: {len(matched)} episodes available, taking {len(pick)}")
        for idx in tqdm(pick, desc=task):
            ep = matched[idx]
            ep_idx, ep_len = ep["episode_index"], ep["length"]
            dc = ep_idx // CHUNKS_SIZE
            t = int(rng.integers(0, ep_len - CHUNK_SIZE + 1))
            pq = hf_hub_download(REPO, f"data/chunk-{dc:03d}/episode_{ep_idx:06d}.parquet", repo_type="dataset")
            df = pd.read_parquet(pq)
            actions = np.stack(df["action"].values[t:t + CHUNK_SIZE]).astype(np.float32).flatten()
            state = np.asarray(df["observation.state"].values[t]).astype(np.float32)
            vid = hf_hub_download(REPO, f"videos/chunk-{dc:03d}/{IMAGE_KEY}/episode_{ep_idx:06d}.mp4", repo_type="dataset")
            buf = io.BytesIO(); decode_frame(vid, t).save(buf, format="PNG")
            rows.append({"task": task, "episode_index": ep_idx, "t": t,
                         "instruction": ep["tasks"][0], "action_chunk": actions,
                         "state": state, "image_png": buf.getvalue()})

    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    print(f"\nwrote {len(out)} contexts across {out['task'].nunique()} tasks -> {args.out}")
    print(out.groupby("task").size().to_string())


if __name__ == "__main__":
    main()
