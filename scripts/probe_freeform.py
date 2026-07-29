#!/usr/bin/env python3
"""Freeform insight loop (user 2026-07-29): let the generalizations emerge from
measured gaps instead of a fixed transform menu.

Round structure (default 4 rounds):
  1. FRESH batch of instructions; Gemini freely rephrases each (6 diverse ways).
  2. Score everything on same-context screens (CRN) -> gaps vs base.
  3. Show Gemini the round's biggest positive/negative gaps + the current
     hypothesis ledger -> it proposes/refines 3-5 GENERALIZATIONS.
  4. Next round ALSO applies each current hypothesis to the fresh batch
     (one variant per hypothesis per instruction) -> out-of-batch test:
     per-hypothesis mean delta with n. Confirmed/refuted lives in the ledger.
Output: results/analysis/freeform_insights.jsonl (one row per round: gaps,
hypotheses, per-hypothesis holdout stats).
  PYTHONPATH=src .venv-gen/bin/python scripts/probe_freeform.py --ipc-dir /workspace/ipc6
"""
import argparse
import json
import os
import time
import uuid
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "src")
from google import genai
from google.genai import types as gtypes

from phrase_rl.gemini_redteam_assets import call_with_retry
from phrase_rl.phase2_train import score_phrases

ap = argparse.ArgumentParser()
ap.add_argument("--ipc-dir", required=True)
ap.add_argument("--out", default="results/analysis/freeform_insights.jsonl")
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--batch", type=int, default=12)
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--screen-f", type=int, default=2)
ap.add_argument("--screen-c", type=int, default=6)
args = ap.parse_args()

sargs = Namespace(k=8, score_seed=0, tau_min=0.0, reward_mode="verifier",
                  k_l2=4, score_timeout=3600, _reward_frames_map=None)
ipc = Path(args.ipc_dir)
if not os.environ.get("GEMINI_API_KEY"):
    for _line in open(os.path.expanduser("~/.bashrc")):
        if _line.startswith("export GEMINI_API_KEY="):
            os.environ["GEMINI_API_KEY"] = _line.split("=", 1)[1].strip().strip('"').strip("'")
            break
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
RNG = np.random.default_rng(67)

ctx = pd.read_parquet("data/contexts_club.parquet")
pool = list(ctx.instruction.unique())
RNG.shuffle(pool)

REPHRASE = """The attached image is a robot arm's camera view (kitchen manipulation). Freely rephrase this instruction {k} DIVERSE ways (same task meaning, any style/structure/vocabulary — be creative and varied):

"{ins}"

Output exactly {k} lines, one rephrasing per line, no numbering, no quotes."""

HYP_APPLY = """The attached image is a robot arm's camera view. Apply this hypothesis about good robot-instruction phrasing to the instruction below (one rewritten instruction, same meaning):

Hypothesis: {hyp}
Instruction: "{ins}"

Output ONE line: the rewritten instruction. No quotes."""

ANALYZE = """We are studying which phrasing properties make a frozen robot policy execute instructions better. Score = gripper-error reward gap vs the base phrasing (positive delta = the rephrasing is BETTER).

This round's largest measured gaps:
BETTER than base:
{pos}
WORSE than base:
{neg}

Current hypothesis ledger (with holdout results so far):
{ledger}

Write 3-5 hypotheses (revise, keep, or replace) that generalize WHY some phrasings work better for this policy. Each must be a one-line actionable rule, testable by rewriting an instruction. Prefer hypotheses that explain multiple gaps. Output one hypothesis per line, no numbering."""


def grips_for(frames, phrases):
    G = []
    for i in range(0, len(frames), 36):
        batch = frames[i:i + 36]
        job = f"ff_{uuid.uuid4().hex[:8]}"
        score_phrases(ipc, job, [(fr, phrases) for fr in batch], sargs)
        G.extend(getattr(score_phrases, "last_grips", None))
    return np.nanmean(np.asarray(G, dtype=float), axis=0)


def frames_for(ins):
    sub = ctx[ctx.instruction == ins]
    eps = sorted(sub.episode_index.unique())
    sc = list(RNG.choice(eps, size=min(args.screen_c, len(eps)), replace=False))
    return (sub[sub.episode_index.isin(sc)].groupby("episode_index").head(args.screen_f).to_dict("records"),
            gtypes.Part.from_bytes(data=bytes(sub.iloc[0].image_png), mime_type="image/png"))


hyps = []  # [{text, n, mean_delta}]
start_round = 0
if os.path.exists(args.out):
    prior = [json.loads(l) for l in open(args.out) if l.strip()]
    if prior:
        hyps = prior[-1].get("ledger_after", [])
        start_round = prior[-1]["round"]
        print(f"resume: round {start_round}, {len(hyps)} hypotheses")

used = start_round * args.batch
for rd in range(start_round + 1, args.rounds + 1):
    t0 = time.time()
    batch = pool[used:used + args.batch]
    used += args.batch
    gaps, hyp_results = [], {h["text"]: [] for h in hyps}
    for ins in batch:
        frames, img = frames_for(ins)
        try:
            out = call_with_retry(client, gtypes, "gemini-3.5-flash",
                                  [img, REPHRASE.format(ins=ins, k=args.k)],
                                  temperature=1.0, max_tokens=600)
        except Exception as e:
            print(f"  gen error: {type(e).__name__}")
            continue
        variants = [l.strip().strip('"').strip() for l in out.splitlines() if l.strip()][:args.k]
        hyp_phr = []
        for h in hyps:
            try:
                hp = call_with_retry(client, gtypes, "gemini-3.5-flash",
                                     [img, HYP_APPLY.format(hyp=h["text"], ins=ins)],
                                     temperature=0.3, max_tokens=100).strip().strip('"').split("\n")[0]
                hyp_phr.append((h["text"], hp))
            except Exception:
                hyp_phr.append((h["text"], None))
        plist = [ins] + variants + [p for _, p in hyp_phr if p]
        g = grips_for(frames, plist)
        base = float(g[0])
        for i, v in enumerate(variants):
            gaps.append({"instruction": ins, "phrase": v, "delta": round(base - float(g[1 + i]), 5)})
        gi = 1 + len(variants)
        for htext, hp in hyp_phr:
            if hp is not None:
                hyp_results[htext].append(base - float(g[gi]))
                gi += 1
    gaps.sort(key=lambda x: -x["delta"])
    pos = "\n".join(f'{x["delta"]:+.4f}  "{x["phrase"][:70]}"  (base: "{x["instruction"][:44]}")' for x in gaps[:10])
    neg = "\n".join(f'{x["delta"]:+.4f}  "{x["phrase"][:70]}"  (base: "{x["instruction"][:44]}")' for x in gaps[-10:])
    for h in hyps:
        res = hyp_results.get(h["text"], [])
        if res:
            h["n"] = h.get("n", 0) + len(res)
            h["mean_delta"] = round(float(np.mean(res + [h.get("mean_delta", 0)] * 0)), 5) if h.get("n", 0) == len(res) \
                else round((h["mean_delta"] * (h["n"] - len(res)) + float(np.sum(res))) / h["n"], 5)
    ledger = "\n".join(f'- {h["text"]}  [holdout: n={h.get("n", 0)}, mean delta {h.get("mean_delta", 0):+.4f}]'
                       for h in hyps) or "(none yet — first round)"
    try:
        out = call_with_retry(client, gtypes, "gemini-3.5-flash",
                              [ANALYZE.format(pos=pos, neg=neg, ledger=ledger)],
                              temperature=0.7, max_tokens=800)
        new_texts = [l.strip().strip('"').strip("-• ").strip() for l in out.splitlines()
                     if l.strip() and len(l.split()) >= 4][:5]
    except Exception as e:
        print(f"  analyze error: {type(e).__name__}")
        new_texts = [h["text"] for h in hyps]
    old = {h["text"]: h for h in hyps}
    hyps = [old.get(t, {"text": t, "n": 0, "mean_delta": 0.0}) for t in new_texts]
    rec = {"round": rd, "batch": batch, "n_gaps": len(gaps),
           "top_gaps": gaps[:10], "bottom_gaps": gaps[-10:],
           "hypothesis_tests": {h: {"n": len(r), "mean_delta": round(float(np.mean(r)), 5) if r else None}
                                for h, r in hyp_results.items()},
           "ledger_after": hyps, "sec": round(time.time() - t0, 1)}
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[round {rd}] {len(gaps)} gaps, {len(hyps)} hypotheses, {rec['sec']:.0f}s", flush=True)
    os.system(f"timeout 300 bash -c 'git add {args.out} && git commit -q -m \"freeform insights round {rd} [pod]\" && git -c rebase.autoStash=true pull -q --rebase && git push -q' >/dev/null 2>&1")

print("FREEFORM-COMPLETE")
