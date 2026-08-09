#!/usr/bin/env python3
"""Cost and resolution of ONE rules-loop evaluation, in the user's parameterisation.

    eval sets: train, train_held (proxy, F x C)   +   sim_held (rollouts)

    proxy_cost = (T_train + T_train_held) * N_task * cost(FxC)
    sim_cost   =  T_sim_held * N_task * num_layouts * K * cost(episode)
    reph_cost  = (T_train + T_train_held + T_sim_held) * N_task * cost(rephraser)

Phrase composition (orig : nat : adv) does not change cost -- every phrase costs
the same to rephrase and score -- but it does change what the number MEANS, so it
is reported.

MEASURED CONSTANTS (this repo, 2026-08-08/09):
  proxy   server bills ~2.7s per context-row for the first phrase and ~0.2s per
          extra phrase riding along (it batches over the phrase list within a
          context). So a task costs  B * (2.7 + 0.2*(P-1))  seconds at F*C=B.
  rollout 0.42 min/episode  (e6 leg: 2304 episodes in ~16h)
  noise   phrase-quality sd 11.9pp (fine_exam, 36 rollouts each);
          within-task proxy logit sd 1.376 (4204 measured training phrases)

    python3 scripts/rules_eval_cost.py
    python3 scripts/rules_eval_cost.py --n-task 12 --fxc 40 --layouts 12 --k 1
"""
import argparse

import numpy as np

SEC_PER_ROW = 2.7          # first phrase on a context row
SEC_PER_EXTRA = 0.2        # each additional phrase on the same row
MIN_PER_EPISODE = 0.42
SD_PHRASE = 0.119          # rollout success sd across phrases, within task
SD_LOGIT = 1.376           # proxy logit sd across phrases, within task
P0 = 0.42                  # base success rate

ap = argparse.ArgumentParser()
ap.add_argument("--t-train", type=int, default=25)
ap.add_argument("--t-train-held", type=int, default=31)
ap.add_argument("--t-sim-held", type=int, default=8)
ap.add_argument("--n-task", type=int, default=8, help="phrases per task (proxy sets)")
ap.add_argument("--n-task-sim", type=int, default=0,
                help="phrases per task on sim_held; 0 = same as --n-task. "
                     "sim_held pools only 8 tasks vs 56 for proxy, so it needs a "
                     "larger N to reach the same resolution -- decoupling buys that "
                     "without paying for proxy scoring nobody asked for.")
ap.add_argument("--fxc", type=int, default=40, help="F*C context-rows per phrase")
ap.add_argument("--layouts", type=int, default=12)
ap.add_argument("--k", type=int, default=1, help="repeats per layout")
ap.add_argument("--orig", type=float, default=0.2)
ap.add_argument("--nat", type=float, default=0.5)
ap.add_argument("--adv", type=float, default=0.3)
ap.add_argument("--iters", type=int, default=6)
ap.add_argument("--rephrasers", type=int, default=1)
ap.add_argument("--reph-sec", type=float, default=2.0, help="sec per rephrase call")
args = ap.parse_args()


def proxy_task_sec(fxc, n_task):
    """One task: fxc context-rows, n_task phrases batched over them."""
    return fxc * (SEC_PER_ROW + SEC_PER_EXTRA * max(n_task - 1, 0))


def resolution(n_proxy_phrases, n_sim_phrases, eps_per_phrase):
    """Detectable paired difference between two rulebooks, 95%."""
    lo = np.log(P0 / (1 - P0))
    d = 1.96 * SD_LOGIT * np.sqrt(2) / np.sqrt(max(n_proxy_phrases, 1))
    proxy_pp = (1 / (1 + np.exp(-(lo + d))) - P0) * 100
    E = n_sim_phrases * eps_per_phrase
    var = 2 * P0 * (1 - P0) / max(E, 1) + 2 * SD_PHRASE ** 2 / max(n_sim_phrases, 1)
    return proxy_pp, 1.96 * np.sqrt(var) * 100


T_tr, T_th, T_sh = args.t_train, args.t_train_held, args.t_sim_held
N = args.n_task
N_sim = args.n_task_sim or N
eps_per_phrase = args.layouts * args.k

proxy_tasks = T_tr + T_th
proxy_sec = proxy_tasks * proxy_task_sec(args.fxc, N)
sim_eps = T_sh * N_sim * eps_per_phrase
sim_h = sim_eps * MIN_PER_EPISODE / 60
reph_sec = ((T_tr + T_th) * N + T_sh * N_sim) * args.reph_sec
per_iter = proxy_sec / 3600 + sim_h + reph_sec / 3600

print("PARAMETERS")
print(f"  tasks           train={T_tr}  train_held={T_th}  sim_held={T_sh}")
print(f"  N_task          proxy {N} / sim {N_sim} phrases per task   composition {args.orig:.0%}/{args.nat:.0%}/{args.adv:.0%} orig/nat/adv")
print(f"  proxy budget    F*C = {args.fxc} context-rows per phrase")
print(f"  sim budget      {args.layouts} layouts x {args.k} rep = {eps_per_phrase} episodes/phrase")
print()
print("PER ITERATION, ONE REPHRASER")
print(f"  proxy (train + train_held) : {proxy_tasks:>4} tasks x {N} phrases  = {proxy_sec/3600:6.2f} GPU-h")
print(f"  rollouts (sim_held)        : {T_sh:>4} tasks x {N_sim} x {eps_per_phrase:<3} = {sim_eps:>6} episodes = {sim_h:6.2f} GPU-h")
print(f"  rephraser calls            : {(T_tr+T_th)*N+T_sh*N_sim:>4} calls                = {reph_sec/3600:6.2f} h (API, not GPU)")
print(f"  {'TOTAL':27}: {per_iter:6.2f} GPU-h")
print()
tot = per_iter * args.iters * args.rephrasers
print(f"FULL RUN  {args.iters} iters x {args.rephrasers} rephraser(s) = {tot:.0f} GPU-h"
      f"  |  1 pod {tot/24:.1f}d   2 pods {tot/2/24:.1f}d   3 pods {tot/3/24:.1f}d")
print()
n_proxy = proxy_tasks * N
n_sim = T_sh * N_sim
ppx, pps = resolution(n_proxy, n_sim, eps_per_phrase)
print("RESOLUTION (detectable difference between two rulebooks, 95%, paired)")
print(f"  proxy  on {n_proxy:>4} phrases                    : {ppx:5.1f} pp")
print(f"  sim    on {n_sim:>4} phrases x {eps_per_phrase} episodes  : {pps:5.1f} pp")
print()
print("  For reference, A28 measured v3-v4 at +1.0 (qwen) / +3.2 (gemini) / +5.2 (claude) pp.")
print("  A resolution above ~5pp cannot see the effect this loop exists to find.")
print()
print("NOTE on layouts vs repeats: paired variance is")
print("    2p(1-p)/E + 2*sd_phrase^2/N_sim      with E = N_sim * layouts * k")
print("  the binomial term depends only on TOTAL episodes, the phrase term only on")
print("  the number of distinct phrases -- so at a fixed episode budget, more")
print("  phrases beats more repeats, and k=1 with many layouts is optimal.")
