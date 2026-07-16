"""Phase 2 reward server: frozen-pi0 CRN scoring over a file-based job queue.

Phase 2 splits across two python environments that cannot be merged: the
trainer runs in .venv-gen (transformers>=5, Qwen3.5 + peft) and the pi0 reward
model only loads in .venv (INTACT-era lerobot fork, transformers 4.48). This
process is the .venv side: it loads the frozen policy ONCE at startup and then
serves scoring jobs through files in --ipc-dir — no sockets, no cross-env
imports.

Protocol (paths inside the req should be absolute):
  trainer writes  <job_id>.req.json
      {"in_parquet": ..., "out_parquet": ..., "k": 16, "seed": 0, "tau_min": 0.25}
  server writes   out_parquet with columns: context_id, phrase,
      loss (mean over the K draws), loss_per_draw (list of K floats)
  server renames  <job_id>.req.json -> <job_id>.done.json  (atomic "ready" signal)
  on error        <job_id>.err.txt (full traceback) + req -> <job_id>.failed.json

in_parquet: one row per (context, phrase) with columns context_id, image_png,
state, action_chunk, phrase; the context fields are taken from each group's
first row. All phrases of a context are scored in ONE Pi0PhraseScorer.score
call so they share the same K (eps, tau) draws — common random numbers, so
within-group advantages are noise-free comparisons. tau_min=0.25 is the
0b-informed setting (see pi0_scoring.make_draws).

The serve loop never dies on a bad job: failures are reported through the
protocol and polling continues. Restart is safe: only the req->done rename
marks completion, so a crash mid-job simply reprocesses that job on the next
start, and out_parquet is written tmp+rename (never left partial). Jobs run in
filename order — the trainer should zero-pad job ids (e.g. step000042).

Usage (on pod, .venv, inside tmux):
  .venv/bin/python -m phrase_rl.phase2_score_server \
      --ipc-dir /workspace/ipc
  # testing: --once processes whatever is pending, then exits
"""

import argparse
import glob
import io
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from phrase_rl.pi0_scoring import Pi0PhraseScorer, import_pi0_policy

DEFAULT_CKPT = "juexzz/INTACT-pi0-finetune-rephrase-bridge"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class Heartbeat:
    """Periodic liveness line for tmux logs; beat() is cheap, call it often."""

    def __init__(self, every: float):
        self.every = every
        self.t0 = self.last = time.time()
        self.n_done = 0
        self.n_failed = 0

    def beat(self):
        now = time.time()
        if now - self.last >= self.every:
            log(f"heartbeat: up {now - self.t0:.0f}s, jobs done={self.n_done} failed={self.n_failed}")
            self.last = now


def process_job(policy, spec, micro_batch, hb, job_id, dim_std=None, ensemble=None):
    """Score one job's parquet; returns (n_contexts, n_rows). Raises on any problem.

    reward_mode="flow" (default): CRN flow-matching residual (Pi0PhraseScorer.score).
    reward_mode="l2": CRN decoded-action per-DoF-normalized L2 (score_l2); needs dim_std.
    reward_mode="verifier": calibrated ensemble logit over pi0 behavioral features
      (loss = -mean member logit; loss_per_draw = per-member -logits). k/seed/tau_min
      are PINNED from the ensemble manifest (must match verifier training slots);
      the job spec's values are ignored in this mode. Needs --stats-contexts and
      --verifier-ensemble at server start.
    All modes share the output schema (loss, loss_per_draw), lower=better, so the
    trainer's advantage/KL/val code is reward-mode-agnostic.
    """
    mode = str(spec.get("reward_mode", "flow"))
    if mode == "verifier":
        if ensemble is None or dim_std is None:
            raise RuntimeError("reward_mode='verifier' needs --verifier-ensemble and --stats-contexts")
        scorer = Pi0PhraseScorer(policy, k=ensemble.k_flow, seed=ensemble.seed,
                                 micro_batch=micro_batch, tau_min=ensemble.tau_min)
    else:
        # per-job (k, seed, tau_min) only re-derives the fixed draws — the policy is shared
        scorer = Pi0PhraseScorer(
            policy,
            k=int(spec.get("k", 16)),
            seed=int(spec.get("seed", 0)),
            micro_batch=micro_batch,
            tau_min=float(spec.get("tau_min", 0.0)),
        )
    if mode == "l2" and dim_std is None:
        raise RuntimeError("reward_mode='l2' requires the server to be started with --stats-contexts")
    k_l2 = int(spec.get("k_l2", 4))
    df = pd.read_parquet(spec["in_parquet"])
    ids, phrases_out, means, per_draw = [], [], [], []
    groups = list(df.groupby("context_id", sort=False))
    for cid, g in tqdm(groups, desc=f"job {job_id} [{mode}]", leave=False):
        row = g.iloc[0]  # context fields identical within a group by construction
        img = np.asarray(Image.open(io.BytesIO(row["image_png"]))).astype(np.float32) / 255.0
        phrases = [str(p) for p in g["phrase"]]
        if mode == "verifier":
            def _frame_logits(fr_row, plist):
                im = np.asarray(Image.open(io.BytesIO(fr_row["image_png"]))).astype(np.float32) / 255.0
                a_st = np.asarray(fr_row["action_chunk"], dtype=np.float32)
                fl, v, u = scorer.score_verbose(
                    im.transpose(2, 0, 1), np.asarray(fr_row["state"]), a_st, plist)
                dec, nl2, grip = scorer.decode_verbose(
                    im.transpose(2, 0, 1), np.asarray(fr_row["state"]), a_st, plist,
                    dim_std, k_l2=ensemble.k_decode)
                feats = pd.DataFrame([{
                    "flow_loss": fl[i].astype(np.float32),
                    "flow_v": v[i].reshape(-1).astype(np.float32),
                    "flow_u": u[i].reshape(-1).astype(np.float32),
                    "decoded": dec[i].reshape(-1).astype(np.float32),
                    "norm_l2": nl2[i].astype(np.float32),
                    "grip_err": grip[i].astype(np.float32),
                    "a_star": a_st.reshape(-1),
                } for i in range(len(plist))])
                return ensemble.member_logits(feats)  # (P, n_members)
            if "frame_id" in g.columns and g["frame_id"].nunique() > 1:
                first = g[g.frame_id == g.frame_id.min()]
                phrases = [str(p) for p in first["phrase"]]
                per_frame = []
                for _, gf in g.groupby("frame_id"):
                    assert [str(p) for p in gf["phrase"]] == phrases, "phrase order mismatch across frames"
                    per_frame.append(_frame_logits(gf.iloc[0], phrases))
                losses = -np.mean(per_frame, axis=0)  # mean calibrated logit across frames
            else:
                losses = -_frame_logits(row, phrases)  # (P, n_members); lower = better
        elif mode == "l2":
            losses = scorer.score_l2(
                img.transpose(2, 0, 1), np.asarray(row["state"]),
                np.asarray(row["action_chunk"]), phrases, dim_std, k_l2=k_l2,
            )  # (P, k_l2), CRN across the whole group
        else:
            losses = scorer.score(
                img.transpose(2, 0, 1), np.asarray(row["state"]),
                np.asarray(row["action_chunk"]), phrases,
            )  # (P, K), CRN across the whole group
        ids.extend([cid] * len(phrases))
        phrases_out.extend(phrases)
        means.extend(losses.mean(axis=1).tolist())
        per_draw.extend([r.tolist() for r in losses])
        hb.beat()

    out = pd.DataFrame(
        {"context_id": ids, "phrase": phrases_out, "loss": means, "loss_per_draw": per_draw}
    )
    out_path = spec["out_parquet"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)  # readers never see a partial parquet
    return len(groups), len(out)


def read_spec(req_path):
    """Parse a req file; returns spec dict, or None if it looks mid-write (retry later)."""
    try:
        with open(req_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # trainer may not write atomically — give a fresh file a few polls to finish
        try:
            if time.time() - os.path.getmtime(req_path) < 5.0:
                return None
        except OSError:
            return None
        raise  # stale and still unparseable -> fail the job through the normal path


def handle_req(policy, req_path, micro_batch, hb, dim_std=None, ensemble=None):
    """Process one req file end to end. Never raises (except KeyboardInterrupt)."""
    base = req_path[: -len(".req.json")]
    job_id = os.path.basename(base)
    try:
        spec = read_spec(req_path)
        if spec is None:
            return  # probably still being written; next poll retries
        t0 = time.time()
        n_ctx, n_rows = process_job(policy, spec, micro_batch, hb, job_id, dim_std=dim_std,
                                    ensemble=ensemble)
        os.replace(req_path, base + ".done.json")  # atomic completion signal
        hb.n_done += 1
        log(f"job {job_id}: {n_ctx} contexts / {n_rows} rows in {time.time() - t0:.1f}s")
    except KeyboardInterrupt:
        raise
    except Exception:
        tb = traceback.format_exc()
        log(f"job {job_id} FAILED:\n{tb}")
        try:  # report through the protocol; a reporting failure must not kill the loop
            with open(base + ".err.txt", "w") as f:
                f.write(tb)
            os.replace(req_path, base + ".failed.json")
            hb.n_failed += 1
        except OSError as e:
            log(f"job {job_id}: could not write failure marker ({e}); will retry job")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipc-dir", required=True)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--poll", type=float, default=0.5, help="seconds between queue scans")
    ap.add_argument("--heartbeat", type=float, default=60.0)
    ap.add_argument("--stats-contexts", default=None,
                    help="parquet whose action_chunk column defines per-DoF L2 normalization (needed for reward_mode=l2)")
    ap.add_argument("--verifier-ensemble", default=None,
                    help="ensemble manifest json (needed for reward_mode=verifier)")
    ap.add_argument("--once", action="store_true", help="process pending jobs, then exit (testing)")
    args = ap.parse_args()

    os.makedirs(args.ipc_dir, exist_ok=True)
    dim_std = None
    if args.stats_contexts:
        chunks = np.stack(pd.read_parquet(args.stats_contexts, columns=["action_chunk"])
                          ["action_chunk"].map(np.asarray))  # (N, H*7)
        dim_std = chunks.reshape(chunks.shape[0], -1, 7).std(axis=(0, 1)) + 1e-8  # (7,) per-DoF std
        log(f"L2 normalization per-DoF std: {np.round(dim_std, 4)}")

    ensemble = None
    if args.verifier_ensemble:
        from phrase_rl.verifier_reward import VerifierEnsemble
        ensemble = VerifierEnsemble(args.verifier_ensemble)
        log(f"verifier ensemble loaded: {len(ensemble.members)} members, "
            f"k_flow={ensemble.k_flow} k_decode={ensemble.k_decode} tau_min={ensemble.tau_min}")

    PI0Policy = import_pi0_policy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading {args.ckpt} on {device} ...")
    policy = PI0Policy.from_pretrained(args.ckpt).to(device)
    log(f"policy loaded; serving {args.ipc_dir}" + (" (--once)" if args.once else ""))

    hb = Heartbeat(args.heartbeat)
    while True:
        reqs = sorted(glob.glob(os.path.join(args.ipc_dir, "*.req.json")))
        for req_path in reqs:
            handle_req(policy, req_path, args.micro_batch, hb, dim_std=dim_std, ensemble=ensemble)
        hb.beat()
        if args.once and not reqs:
            log(f"--once: queue drained ({hb.n_done} done, {hb.n_failed} failed); exiting")
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
