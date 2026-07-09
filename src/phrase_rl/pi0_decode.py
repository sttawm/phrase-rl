"""Decode pi0 actions per (context, phrase) with common random numbers — bake-off infra.

For each matched 0c context and each phrase: run pi0's full denoising
(model.sample_actions) K times with FIXED noise draws shared across phrases,
producing the action chunk the policy would execute. Outputs decoded chunks +
per-dimension-normalized L2 vs the ground-truth chunk (normalization: per-dim std
over the train contexts' chunks; gripper dim reported separately).

Feeds bake-off arms: decoded-L2 reward, and (later) CoVer-verifier scoring of
(original instruction, decoded action).

Run on pod (.venv):
  python -m phrase_rl.pi0_decode --contexts data/contexts_0c_match.parquet \
    --phrases data/phrases_0c.parquet --stats-contexts data/contexts_train.parquet \
    --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --out data/decoded_0c.parquet
"""

import argparse
import io
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from phrase_rl.pi0_scoring import import_pi0_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--phrases", required=True)
    ap.add_argument("--stats-contexts", required=True, help="parquet whose action_chunk column defines per-dim normalization")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4, help="decode samples per phrase (shared noise)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stats = pd.read_parquet(args.stats_contexts, columns=["action_chunk"])
    chunks = np.stack(stats["action_chunk"].map(np.asarray))  # (N, 28)
    dim_std = chunks.reshape(-1, 4, 7).std(axis=(0, 1)) + 1e-8  # (7,) per-DoF std
    print("per-dim std:", np.round(dim_std, 4))

    PI0Policy = import_pi0_policy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PI0Policy.from_pretrained(args.ckpt).to(device).eval()
    model = policy.model
    cfg = policy.config
    horizon = cfg.chunk_size
    padded = getattr(cfg, "max_action_dim", 7)
    g = torch.Generator().manual_seed(args.seed)
    fixed_noise = torch.randn(args.k, horizon, padded, generator=g)  # shared across phrases

    contexts = pd.read_parquet(args.contexts)
    phrases = pd.read_parquet(args.phrases)
    done = []
    keys = set()
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        done = prev.to_dict("records")
        keys = {(r["task"], r["episode_index"], r["t"], r["phrase"], r["k"]) for r in done}
        print(f"resume: {len(done)} rows")

    tasks = [t for t in phrases["task"].unique() if t in set(contexts["task"])]
    for task in tasks:
        plist = list(dict.fromkeys(phrases[phrases.task == task]["phrase"]))
        for row in tqdm(list(contexts[contexts.task == task].itertuples()), desc=task):
            a_star = np.asarray(row.action_chunk, dtype=np.float32).reshape(horizon, 7)
            img = np.asarray(Image.open(io.BytesIO(row.image_png))).astype(np.float32) / 255.0
            img_t = torch.from_numpy(img.transpose(2, 0, 1))[None].to(device)
            st = torch.from_numpy(np.asarray(row.state, dtype=np.float32))[None].to(device)
            for ki in range(args.k):
                todo = [p for p in plist if (task, row.episode_index, row.t, p, ki) not in keys]
                if not todo:
                    continue
                with torch.no_grad():
                    for p in todo:
                        batch = {"observation.state": st, "task": [p]}
                        for key in cfg.image_features:
                            batch[key] = img_t
                        norm = policy.normalize_inputs(batch)
                        images, img_masks = policy.prepare_images(norm)
                        state = policy.prepare_state(norm)
                        lang_tokens, lang_masks = policy.prepare_language(norm)
                        acts = model.sample_actions(
                            images, img_masks, lang_tokens, lang_masks, state,
                            noise=fixed_noise[ki][None].to(device),
                        )  # (1, horizon, padded)
                        acts = policy.unnormalize_outputs({"action": acts[:, :, :7]})["action"]
                        dec = acts[0].float().cpu().numpy()  # (horizon, 7)
                        nl2 = float(np.sqrt((((dec - a_star) / dim_std) ** 2)[:, :6].mean()))
                        grip = float(np.abs((dec - a_star))[:, 6].mean())
                        done.append({"task": task, "episode_index": row.episode_index,
                                     "t": row.t, "phrase": p, "k": ki,
                                     "decoded": dec.flatten(), "norm_l2": nl2, "grip_err": grip})
            pd.DataFrame(done).to_parquet(args.out, index=False)
    print(f"wrote {len(done)} rows -> {args.out}")


if __name__ == "__main__":
    main()
