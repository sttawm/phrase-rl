"""Phase 0a smoke test: load INTACT pi0, verify CRN scoring end to end.

Checks, in order:
 1. Policy loads; forward() accepts noise/time kwargs (CRN hook exists).
 2. Scoring a real Bridge context runs and returns finite losses.
 3. Determinism: same seed -> bit-identical losses (CRN foundation).
 4. Sanity: prints losses for original / paraphrase / wrong-object / gibberish —
    a first qualitative peek at phrase sensitivity, NOT a gate.

Usage (on pod):
  python -m phrase_rl.smoke_pi0 --contexts data/contexts_val_0b.parquet
"""

import argparse
import inspect
import io

import numpy as np
import pandas as pd
import torch
from PIL import Image

from phrase_rl.pi0_scoring import Pi0PhraseScorer

DEFAULT_CKPT = "juexzz/INTACT-pi0-finetune-rephrase-bridge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    import lerobot
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

    print(f"lerobot {getattr(lerobot, '__version__', '?')}, torch {torch.__version__}")

    sig = inspect.signature(PI0Policy.forward)
    assert "noise" in sig.parameters and "time" in sig.parameters, (
        f"PI0Policy.forward has no noise/time kwargs (params: {list(sig.parameters)}); "
        "CRN hook needs a different lerobot pin or a monkey-patch"
    )
    print("[1] forward() accepts noise/time — CRN hook OK")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PI0Policy.from_pretrained(args.ckpt).to(device)
    print(f"[1] loaded {args.ckpt} on {device}")
    print(f"    image_features={list(policy.config.image_features)}")
    print(f"    chunk_size={policy.config.chunk_size}, action_dim={policy.config.action_feature.shape}")

    row = pd.read_parquet(args.contexts).iloc[0]
    img = np.asarray(Image.open(io.BytesIO(row["image_png"]))).astype(np.float32) / 255.0
    img_chw = img.transpose(2, 0, 1)
    instruction = row["instruction"]
    print(f"[2] context: ep={row['episode_index']} t={row['t']} instr={instruction!r}")

    phrases = [
        instruction,
        f"please {instruction}",
        "move the coffee mug onto the top shelf",
        "flibber jabberwock quantum banana",
    ]
    scorer = Pi0PhraseScorer(policy, k=args.k, seed=0)
    losses = scorer.score(img_chw, np.asarray(row["state"]), np.asarray(row["action_chunk"]), phrases)
    assert np.isfinite(losses).all(), f"non-finite losses: {losses}"
    print(f"[2] losses finite, shape={losses.shape}")

    scorer2 = Pi0PhraseScorer(policy, k=args.k, seed=0)
    losses2 = scorer2.score(img_chw, np.asarray(row["state"]), np.asarray(row["action_chunk"]), phrases)
    assert np.array_equal(losses, losses2), "same seed gave different losses — CRN broken"
    print("[3] determinism OK (same seed -> identical losses)")

    print("[4] mean loss per phrase (K draws, shared):")
    for phrase, l in zip(phrases, losses.mean(axis=1)):
        print(f"    {l:8.5f}  {phrase!r}")


if __name__ == "__main__":
    main()
