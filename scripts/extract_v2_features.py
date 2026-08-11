#!/usr/bin/env python3
"""Pod-side: dump per-frame verifier-v2 features for the 434 gt phrases.

For every (task, phrase) in the worklist and every context frame of its task,
write one row with the frozen ensemble's 64-d penultimate embedding (mean over
members), the per-member calibrated logits, the grip error, and t -- the
inputs the verifier-v2 residual head trains on. Mirrors the score server's
_frame_feats assembly exactly (phase2_score_server.py); the ensemble stays
frozen -- this is feature EXTRACTION, not a change to the deployed reward.

Needs the same context tables as bank scoring (e6 has all 15 tasks; e4 covers
11 via the restored contexts_sim_val8). Run on a GPU pod:

  PYTHONPATH=/workspace/phrase-rl/src /workspace/INT-ACT/.venv/bin/python \
    scripts/extract_v2_features.py --out results/analysis/v2_features_gt434.parquet

Resumable on (task, phrase); flushes per phrase. Batches all of a task's
worklist phrases through each frame together (the server's economics: the
policy pass is per-frame, phrases ride along nearly free).
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/workspace/phrase-rl")
sys.path.insert(0, str(REPO / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", default="results/analysis/selfagree_worklist.parquet")
    ap.add_argument("--out", default="results/analysis/v2_features_gt434.parquet")
    ap.add_argument("--ensemble", default="results/checkpoints/verifier_reward_ensemble_4f.json")
    ap.add_argument("--stats-contexts", default="results/phrase_artifacts/chunk_stats.parquet")
    ap.add_argument("--ckpt", default="juexzz/INTACT-pi0-finetune-rephrase-bridge")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from phrase_rl.verifier_reward import VerifierEnsemble
    from phrase_rl.pi0_scoring import Pi0PhraseScorer
    from phrase_rl.phase2_score_server import import_pi0_policy

    ensemble = VerifierEnsemble(str(REPO / args.ensemble))
    chunks = np.stack(pd.read_parquet(REPO / args.stats_contexts, columns=["action_chunk"])
                      ["action_chunk"].map(np.asarray))
    dim_std = chunks.reshape(chunks.shape[0], -1, 7).std(axis=(0, 1)) + 1e-8

    PI0Policy = import_pi0_policy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PI0Policy.from_pretrained(args.ckpt).to(device)
    scorer = Pi0PhraseScorer(policy, k=ensemble.k_flow, seed=ensemble.seed,
                             tau_min=ensemble.tau_min)

    work = pd.read_parquet(REPO / args.worklist)
    parts = []
    for f in ["data/contexts_sim_oov.parquet", "data/contexts_sim_oov5.parquet",
              "data/contexts_sim_val8.parquet"]:
        p = REPO / f
        if p.exists():
            d = pd.read_parquet(p)
            k = "task" if "task" in d.columns else "instruction"
            d["_key"] = d[k].astype(str)
            parts.append(d)
            print(f"contexts: {f} ({len(d)} rows)", flush=True)
    ctx = pd.concat(parts, ignore_index=True)

    out_path = REPO / args.out
    done, rows = set(), []
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        rows = prev.to_dict("records")
        done = set(zip(prev.task, prev.phrase))
        print(f"resume: {len(done)} (task, phrase) done", flush=True)

    for task, grp in work.groupby("task"):
        t = str(task)
        sub = ctx[ctx._key.isin([t, t[len("widowx_"):] if t.startswith("widowx_") else t])]
        phrases = [p for p in dict.fromkeys(grp.phrase.astype(str)) if (task, p) not in done]
        if len(sub) == 0 or not phrases:
            if not len(sub):
                print(f"{t}: NO CONTEXTS", flush=True)
            continue
        for _, fr in sub.iterrows():
            im = np.asarray(Image.open(io.BytesIO(fr["image_png"]))).astype(np.float32) / 255.0
            a_st = np.asarray(fr["action_chunk"], dtype=np.float32)
            fl, v, u = scorer.score_verbose(
                im.transpose(2, 0, 1), np.asarray(fr["state"]), a_st, phrases)
            dec, nl2, grip = scorer.decode_verbose(
                im.transpose(2, 0, 1), np.asarray(fr["state"]), a_st, phrases,
                dim_std, k_l2=ensemble.k_decode)
            feats = pd.DataFrame([{
                "flow_loss": fl[i].astype(np.float32),
                "flow_v": v[i].reshape(-1).astype(np.float32),
                "flow_u": u[i].reshape(-1).astype(np.float32),
                "decoded": dec[i].reshape(-1).astype(np.float32),
                "norm_l2": nl2[i].astype(np.float32),
                "grip_err": grip[i].astype(np.float32),
                "a_star": a_st.reshape(-1),
            } for i in range(len(phrases))])
            emb = ensemble.member_embeddings(feats)          # (P, 64)
            lg = ensemble.member_logits(feats)               # (P, members)
            for i, p in enumerate(phrases):
                rows.append({"task": task, "phrase": p,
                             "episode_index": int(fr["episode_index"]), "t": int(fr["t"]),
                             "embed": emb[i].astype(np.float32),
                             "member_logits": lg[i].astype(np.float32),
                             "grip_row": float(np.mean(grip[i]))})
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"[{t}] {len(phrases)} phrases x {len(sub)} frames -> {len(rows)} rows",
              flush=True)
    print("EXTRACT-V2-DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
