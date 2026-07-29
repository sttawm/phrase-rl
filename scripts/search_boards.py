#!/usr/bin/env python3
"""Oracle-style iterative board search over training instructions (user spec
2026-07-29): per instruction — 16-board incl. original, rank, regenerate
informed by the board keeping top-4, repeat to convergence, then confirm the
final top-4 + original at full fidelity. Depth-first: one instruction completes
before the next starts; output streams to results/analysis/search_boards.jsonl.

Cost design: iterative rounds rank on a SCREEN cell (F=1 x C=4, ~6s/phrase);
only the final confirmation uses F=4 x C=10. Gemini = flash, 1 image/call.
Run on pod6 (score server + GEMINI_API_KEY):
  PYTHONPATH=src .venv-gen/bin/python scripts/search_boards.py --ipc-dir /workspace/ipc6
"""
import argparse
import io
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
ap.add_argument("--out", default="results/analysis/search_boards.jsonl")
ap.add_argument("--board", type=int, default=16)
ap.add_argument("--keep", type=int, default=4)
ap.add_argument("--max-rounds", type=int, default=4)
ap.add_argument("--min-gain", type=float, default=0.002)
ap.add_argument("--n-instructions", type=int, default=100)
ap.add_argument("--screen-f", type=int, default=1, help="frames/episode in iterative rounds")
ap.add_argument("--screen-c", type=int, default=4, help="episodes in iterative rounds")
args = ap.parse_args()

sargs = Namespace(k=8, score_seed=0, tau_min=0.0, reward_mode="verifier",
                  k_l2=4, score_timeout=3600, _reward_frames_map=None)
ipc = Path(args.ipc_dir)
client = genai.Client()
RNG = np.random.default_rng(23)

ctx = pd.read_parquet("data/contexts_club.parquet")
gem_seed = {}
try:  # optional round-1 seeds from the (retired) bulk generator; absence is fine
    cand = pd.read_parquet("results/analysis/search_candidates.parquet")
    gem_seed = {i: list(s[s.source == "gemini"].phrase)
                for i, s in cand.groupby("instruction")}
except Exception:
    pass
sizes = ctx.groupby("instruction").episode_index.nunique()
order = list(sizes.sort_values(ascending=False).index)[:args.n_instructions]

GEN_PROMPT = """The attached image is a robot arm's camera view (BridgeData kitchen manipulation). We are searching for the instruction phrasing a trained robot policy follows most reliably.

Task instruction (original): "{nominal}"

Ranked results so far (LOWER score = better; gripper-error reward):
{board}

Write {n} NEW phrasings informed by what is winning: keep the elements that work (object names, adjectives, structure), vary what might improve. Use common household object names matching the image; concrete visible colors; imperative; under 15 words; same task meaning. No duplicates of the listed phrases.
Output exactly {n} lines, one per line, no numbering, no quotes."""


def parse_lines(out, seen, n):
    res = []
    for line in out.splitlines():
        p = line.strip().strip('"').strip("-• ").strip()
        if p and 3 <= len(p.split()) <= 18 and p.lower() not in seen and len(res) < n:
            seen.add(p.lower())
            res.append(p)
    return res


def grips_for(frames, phrases):
    G = []
    for i in range(0, len(frames), 36):
        batch = frames[i:i + 36]
        job = f"brd_{uuid.uuid4().hex[:8]}"
        score_phrases(ipc, job, [(fr, phrases) for fr in batch], sargs)
        G.extend(getattr(score_phrases, "last_grips", None))
    return np.nanmean(np.asarray(G, dtype=float), axis=0)


done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["instruction"] for l in open(args.out)}
    print(f"resume: {len(done)} instructions complete")

for ins in order:
    if ins in done:
        continue
    t0 = time.time()
    sub = ctx[ctx.instruction == ins]
    eps = sorted(sub.episode_index.unique())
    sc_eps = list(RNG.choice(eps, size=min(args.screen_c, len(eps)), replace=False))
    fin_eps = list(RNG.choice(eps, size=min(10, len(eps)), replace=False))
    sc_frames = sub[sub.episode_index.isin(sc_eps)].groupby("episode_index").head(args.screen_f).to_dict("records")
    fin_frames = sub[sub.episode_index.isin(fin_eps)].to_dict("records")
    img = gtypes.Part.from_bytes(data=bytes(sub.iloc[0].image_png), mime_type="image/png")

    seen = {ins.strip().lower()}
    board = [ins] + parse_lines("\n".join(gem_seed.get(ins, [])), seen, args.board - 1)
    if len(board) < args.board:  # top up round-1 to a full board
        try:
            out = call_with_retry(client, gtypes, "gemini-3.5-flash",
                                  [img, GEN_PROMPT.format(nominal=ins, n=args.board - len(board),
                                                          board="(no results yet — first round)")],
                                  temperature=0.9, max_tokens=700)
            board += parse_lines(out, seen, args.board - len(board))
        except Exception as e:
            print(f"  gen error r1: {type(e).__name__}")

    scored = {}  # phrase -> screen grip (same contexts all rounds -> comparable)
    g = grips_for(sc_frames, board)
    scored.update({p: float(v) for p, v in zip(board, g)})
    rounds, best_hist = [], []
    for rd in range(1, args.max_rounds + 1):
        ranked = sorted(scored.items(), key=lambda x: x[1])
        top = ranked[:args.keep]
        best_hist.append(top[0][1])
        rounds.append({"round": rd, "board_size": len(scored),
                       "top": [{"phrase": p, "screen_grip": round(v, 5)} for p, v in top]})
        if rd >= 2 and best_hist[-2] - best_hist[-1] < args.min_gain:
            break  # converged
        if rd == args.max_rounds:
            break
        btxt = "\n".join(f'{i+1}. "{p}"  score={v:.4f}' for i, (p, v) in enumerate(ranked[:8]))
        try:
            out = call_with_retry(client, gtypes, "gemini-3.5-flash",
                                  [img, GEN_PROMPT.format(nominal=ins, n=args.board - args.keep, board=btxt)],
                                  temperature=0.9, max_tokens=700)
        except Exception as e:
            print(f"  gen error r{rd+1}: {type(e).__name__}")
            break
        newp = parse_lines(out, seen, args.board - args.keep)
        if not newp:
            break
        g = grips_for(sc_frames, newp)
        scored.update({p: float(v) for p, v in zip(newp, g)})

    finalists = [p for p, _ in sorted(scored.items(), key=lambda x: x[1])[:args.keep]]
    if ins not in finalists:
        finalists.append(ins)
    gf = grips_for(fin_frames, finalists)
    franked = sorted(zip(finalists, gf), key=lambda x: x[1])
    winner, wg = franked[0]
    orig_g = dict(franked).get(ins, float("nan"))
    rec = {"instruction": ins, "club_eps": len(eps), "n_scored": len(scored),
           "rounds": rounds, "screen_eps": [int(e) for e in sc_eps],
           "finals_eps": [int(e) for e in fin_eps],
           "final_board": [{"phrase": p, "grip": round(float(v), 5)} for p, v in franked],
           "winner": winner, "winner_grip": round(float(wg), 5),
           "orig_grip": round(float(orig_g), 5),
           "delta": round(float(orig_g - wg), 5),
           "winner_is_original": winner == ins,
           "sec": round(time.time() - t0, 1)}
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    done.add(ins)
    print(f"[{len(done)}/{len(order)}] {ins[:44]!r} rounds={len(rounds)} scored={len(scored)} "
          f"delta={rec['delta']:.4f} {'(orig wins)' if rec['winner_is_original'] else ''} {rec['sec']:.0f}s", flush=True)
    if len(done) % 5 == 0:
        os.system(f"timeout 300 bash -c 'git add {args.out} && git commit -q -m \"board search progress [pod]\" && git -c rebase.autoStash=true pull -q --rebase && git push -q' >/dev/null 2>&1")

os.system(f"timeout 300 bash -c 'git add {args.out} && git commit -q -m \"board search complete [pod]\" && git -c rebase.autoStash=true pull -q --rebase && git push -q' >/dev/null 2>&1")
print("BOARDS-COMPLETE")
