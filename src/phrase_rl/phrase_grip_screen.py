"""Corpus-scale phrase-edit screening via the exam-validated grip channel.

For each Bridge episode context (real training frames + its GT instruction),
apply text-only edit transforms, score all variants with CRN-paired pi0
decodes (decode_verbose), and report the PAIRED per-context grip delta per
edit across episodes. DISCOVERY instrument only: signs are validated against
the certified rollout cells (clause family x8, six-edit matrix); rollout
certification remains ground truth (EXPERIMENT.md 2026-07-23 grip-screen).

Lower grip_err = closer to the demo gripper trajectory = "better" under the
screen. A NEGATIVE mean delta for an edit means the edit IMPROVED (reduced)
grip error vs the base instruction.

  .venv-gen/bin/python -m phrase_rl.phrase_grip_screen \
      --contexts data/contexts_train_multit16.parquet \
      --stats-contexts results/phrase_artifacts/chunk_stats.parquet \
      --episodes 200 --frames 4 --k 4 --out results/analysis/grip_screen_pilot.json
"""
import argparse
import io
import json
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image

from phrase_rl.pi0_scoring import Pi0PhraseScorer, import_pi0_policy

CATEGORY_MAP = {
    "carrot": "vegetable", "eggplant": "vegetable", "corn": "vegetable",
    "banana": "fruit", "spoon": "utensil", "fork": "utensil", "knife": "utensil",
    "cup": "container", "mug": "container", "bowl": "container", "pot": "container",
    "can": "object", "bottle": "object", "block": "object", "cloth": "fabric",
    "towel": "fabric", "sushi": "food", "mushroom": "vegetable",
}


def edits_for(instr: str):
    """Yield (edit_name, variant) for every transform applicable to instr."""
    out = {}
    low = instr.lower().strip()
    if re.search(r"\bput\b", low):
        out["put_to_set"] = re.sub(r"\bput\b", "set", low, count=1)
        out["put_to_place"] = re.sub(r"\bput\b", "place", low, count=1)
    if re.search(r"\bon\b(?! top)", low):
        out["on_to_onto"] = re.sub(r"\bon\b(?! top)", "onto", low, count=1)
        out["on_to_on_top_of"] = re.sub(r"\bon\b(?! top)", "on top of", low, count=1)
    m = re.match(r"^put (the )?(?P<src>.+?) (?P<rel>on|in|into|onto) (?P<dst>.+)$", low)
    if m:
        out["clause_wrap"] = (f"pick up the {m.group('src')} and put it "
                              f"{m.group('rel')} {m.group('dst')}")
    tele = re.sub(r"\b(the|a|an)\b ", "", low)
    if tele != low:
        out["telegram"] = re.sub(r"\s+", " ", tele).strip()
    out["please"] = f"please {low}"
    for noun, cat in CATEGORY_MAP.items():
        if re.search(rf"\b{noun}\b", low):
            out["category_noun"] = re.sub(rf"\b{noun}\b", cat, low, count=1)
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--stats-contexts", required=True)
    ap.add_argument("--ckpt", default="juexzz/INTACT-pi0-finetune-rephrase-bridge")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--frames", type=int, default=4, help="frames per episode (linspace subsample)")
    ap.add_argument("--k", type=int, default=4, help="CRN decode draws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    chunks = np.stack(pd.read_parquet(args.stats_contexts, columns=["action_chunk"])
                      ["action_chunk"].map(np.asarray))
    dim_std = chunks.reshape(chunks.shape[0], -1, 7).std(axis=(0, 1)) + 1e-8

    df = pd.read_parquet(args.contexts)
    eps = sorted(df.episode_index.unique())
    rng = np.random.default_rng(args.seed)
    eps = list(rng.choice(eps, size=min(args.episodes, len(eps)), replace=False))
    df = df[df.episode_index.isin(eps)]

    PI0Policy = import_pi0_policy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PI0Policy.from_pretrained(args.ckpt).to(device).eval()
    scorer = Pi0PhraseScorer(policy, k=8, seed=args.seed, micro_batch=64)

    per_edit = {}   # edit -> list of per-episode mean paired deltas
    n_ctx = 0
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("t").reset_index(drop=True)
        idx = np.unique(np.linspace(0, len(g) - 1, args.frames).round().astype(int))
        g = g.iloc[idx]
        instr = str(g.iloc[0]["instruction"]).strip()
        variants = edits_for(instr)
        if not variants:
            continue
        phrases = [instr] + list(variants.values())
        names = list(variants.keys())
        frame_deltas = {nm: [] for nm in names}
        ok = True
        for _, row in g.iterrows():
            img = np.asarray(Image.open(io.BytesIO(row["image_png"]))).astype(np.float32) / 255.0
            try:
                _, _, grip = scorer.decode_verbose(
                    img.transpose(2, 0, 1), np.asarray(row["state"], dtype=np.float32),
                    np.asarray(row["action_chunk"], dtype=np.float32), phrases,
                    dim_std, k_l2=args.k)
            except Exception as e:  # noqa: BLE001 — skip broken context, keep the sweep
                print(f"[skip] ep{ep} t{row['t']}: {e}", flush=True)
                ok = False
                break
            base = grip[0].mean()
            for i, nm in enumerate(names):
                frame_deltas[nm].append(float(grip[i + 1].mean() - base))
        if not ok:
            continue
        n_ctx += 1
        for nm in names:
            per_edit.setdefault(nm, []).append(float(np.mean(frame_deltas[nm])))
        if n_ctx % 20 == 0:
            print(f"[{n_ctx}] episodes screened", flush=True)

    rows = []
    for nm, deltas in sorted(per_edit.items()):
        d = np.asarray(deltas)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        rows.append({"edit": nm, "n_episodes": len(d),
                     "mean_delta_grip": round(float(d.mean()), 6),
                     "se": round(float(se), 6),
                     "z": round(float(d.mean() / se), 2) if se and not np.isnan(se) else None,
                     "frac_improved": round(float((d < 0).mean()), 3)})
    out = {"contexts": args.contexts, "episodes_screened": n_ctx,
           "frames_per_ep": args.frames, "k": args.k,
           "note": "negative mean_delta_grip = edit reduced grip error (screen-better)",
           "table": rows}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n{'edit':18s} {'n':>5s} {'mean d(grip)':>13s} {'z':>7s} {'frac<0':>7s}")
    for r in rows:
        print(f"{r['edit']:18s} {r['n_episodes']:5d} {r['mean_delta_grip']:13.6f} "
              f"{str(r['z']):>7s} {r['frac_improved']:7.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
