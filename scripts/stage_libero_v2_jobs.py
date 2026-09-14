#!/usr/bin/env python3
"""Shard the sealed-v2 round-1 rollouts into git job-queue legs (pi0.5 / LIBERO).

Every arm's rewrites plus the un-rephrased naturals collapse to a set of unique
(task, phrase) pairs -- identical strings are rolled ONCE and shared across arms
(PREREG: key is (task, phrase, init)). Round 1 rolls ALL 50 init states, 1 rep.
Legs are balanced by episode count with task affinity (bank_eval reloads the sim
scene per task); a task is split across legs only when it alone exceeds 1.5x the
target leg size, and then by phrase. Written to results/rules_runs/<run>/jobs/ in
exactly the shape rules_loop_jobs.py's libero_bank_eval branch consumes:
  <jid>.payload.parquet  columns task ("suite:task_id"), phrase
  <jid>.spec.json        {"job_id", "kind": "score", "method": "libero_bank_eval",
                          "prereg", "rollout": {"inits", "port", "seed"}}
Task keys are accepted as "suite:task_id" or "suite/task_id" (the analysis-file
convention) and always written as "suite:task_id". Phrase values are staged
verbatim (only whitespace-empty strings are dropped), so payload/result phrases
join exactly back to naturals_v2.parquet and the apply parquets.

Idempotent per leg: a leg whose content hash already has a spec on disk is left
alone. Because ANY input change re-partitions the round and gives every leg a
new hash, the script refuses to write when a new leg's (task, phrase) pairs are
already staged in the run dir (the worker would roll them twice); pass
--allow-overlap only when that is intended. The apply arms are part of the round,
so a missing or empty --applies-dir is an error outside --dry-run.

  FINAL_EVAL=1 .venv/bin/python scripts/stage_libero_v2_jobs.py --legs 48
  FINAL_EVAL=1 .venv/bin/python scripts/stage_libero_v2_jobs.py --dry-run
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import re

import numpy as np
import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

REPO = pathlib.Path(__file__).resolve().parents[1]
D = REPO / "results/analysis/pi05_bank"
INITS = list(range(50))              # round 1: every LIBERO init state, 1 rep
SPLIT_FACTOR = 1.5                   # a task is split only above this x target
TASK_RE = re.compile(r"^libero_(spatial|object|goal|10|90)[:/]\d+$")

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--naturals", default=str(D / "naturals_v2.parquet"),
                help="bases parquet (task, phrase)")
ap.add_argument("--applies-dir", default=str(D / "eval_applies_v2"),
                help="dir of *.parquet arms with columns task, phrase, rewrite")
ap.add_argument("--legs", type=int, default=48)
ap.add_argument("--run", default="p_v2")
ap.add_argument("--prefix", default="v2r1")
ap.add_argument("--prereg", default="pi05 sealed v2 round-1: all 50 inits x 1 rep",
                help="short provenance string copied into every spec")
ap.add_argument("--dry-run", action="store_true",
                help="print the leg table only; write nothing")
ap.add_argument("--allow-overlap", action="store_true",
                help="write even if new legs repeat (task,phrase) pairs already "
                     "staged in the run dir (they WILL be rolled twice)")
a = ap.parse_args()


def norm_task(t):
    """'suite/task_id' (analysis-file form) -> 'suite:task_id' (queue form)."""
    return str(t).strip().replace("/", ":", 1)


def load_pairs():
    """All (task, phrase, src) rows across the bases and every apply arm."""
    nat = pathlib.Path(a.naturals)
    if nat.exists():
        bases = pd.read_parquet(nat)
    elif a.dry_run:
        print(f"NOTE: {nat} missing -> synthesizing 20 tasks x 10 phrases (dry-run)")
        bases = pd.DataFrame([{"task": f"libero_90/{t}", "phrase": f"fake phrase {t}-{k}"}
                              for t in range(20) for k in range(10)])
    else:
        raise SystemExit(f"missing bases: {nat}")
    pairs = [bases[["task", "phrase"]].assign(src="base")]
    adir = pathlib.Path(a.applies_dir)
    files = sorted(adir.glob("*.parquet")) if adir.is_dir() else []
    if not files:
        # the arms are part of the round: staging bases alone would leave them
        # to be re-partitioned (new hashes) beside these specs later
        if not a.dry_run:
            raise SystemExit(f"no apply arms (*.parquet) under {adir}; refusing to "
                             f"stage bases alone (use --dry-run to preview)")
        print(f"NOTE: no apply arms under {adir}; dry-run table covers bases only")
    for f in files:
        d = pd.read_parquet(f)
        d = d[d.rewrite.fillna("").astype(str).str.strip().str.len() > 0]
        pairs.append(d[["task", "rewrite"]].rename(columns={"rewrite": "phrase"})
                     .assign(src=f.stem))
    allp = pd.concat(pairs, ignore_index=True)
    allp["task"] = allp.task.astype(str)
    bad = sorted(set(t for t in allp.task if not TASK_RE.match(t.strip())))
    if bad:
        raise SystemExit(f"task keys must be 'suite:task_id' or 'suite/task_id', "
                         f"got e.g. {bad[:5]}")
    allp["task"] = allp.task.map(norm_task)
    # keep the phrase VALUE verbatim (exact join key back to the source files);
    # strip only to decide emptiness
    allp["phrase"] = allp.phrase.astype(str)
    return allp[allp.phrase.str.strip().str.len() > 0].reset_index(drop=True)


def make_units(uniq, n_legs):
    """Task-affinity units: (label, task, phrase-rows). Oversized tasks are cut
    into near-equal phrase chunks so no single unit dwarfs the target leg."""
    target = len(uniq) * len(INITS) / n_legs
    units = []
    for task, g in uniq.groupby("task", sort=False):
        g = g.sort_values("phrase").reset_index(drop=True)
        eps = len(g) * len(INITS)
        if eps > SPLIT_FACTOR * target and len(g) > 1:
            k = min(len(g), math.ceil(eps / target))
            for j, idx in enumerate(np.array_split(np.arange(len(g)), k)):
                units.append((f"{task}#{j}", task, g.iloc[idx].reset_index(drop=True)))
        else:
            units.append((task, task, g))
    # LPT: biggest unit first onto the lightest leg
    units.sort(key=lambda u: -len(u[2]))
    legs = [[] for _ in range(n_legs)]
    load = [0] * n_legs
    for u in units:
        i = load.index(min(load))
        legs[i].append(u)
        load[i] += len(u[2])
    return legs, target


def content_hash(pl):
    return hashlib.sha1(("|".join(sorted(f"{t}\x01{p}" for t, p in
                                         zip(pl.task, pl.phrase)))).encode()
                        ).hexdigest()[:10]


def staged_pairs(jd):
    """(task, phrase) pairs already in any payload of the run dir, by leg hash."""
    out = {}
    for f in sorted(jd.glob("*.payload.parquet")) if jd.exists() else []:
        h = f.name[:-len(".payload.parquet")].rsplit("_", 1)[-1]
        p = pd.read_parquet(f)
        out[h] = set(zip(p.task.astype(str), p.phrase.astype(str)))
    return out


allp = load_pairs()
uniq = allp[["task", "phrase"]].drop_duplicates().reset_index(drop=True)
print(f"{len(allp)} arm rows -> {len(uniq)} unique (task,phrase) over "
      f"{uniq.task.nunique()} tasks; {len(uniq)*len(INITS)} episodes")

# dedup ledger: which arms share which string (keeps arm composition auditable)
ledger = (allp.groupby(["task", "phrase"]).src.apply(lambda s: sorted(set(s)))
          .reset_index().rename(columns={"src": "arms"}))
ledger["n_arms"] = ledger.arms.str.len()
print(f"dedup ledger: {int((ledger.n_arms > 1).sum())} strings shared by >1 arm")

# preflight: every kept line, before anything is written (sealed-set discipline)
for r in uniq.sort_values(["task", "phrase"]).itertuples():
    print(f"  PREFLIGHT {r.task}: {r.phrase!r}")

legs, target = make_units(uniq, a.legs)
split_tasks = sorted({u[1] for leg in legs for u in leg if "#" in u[0]})
print(f"target leg size {target:.0f} episodes; "
      f"{len(split_tasks)} task(s) split by phrase: {split_tasks}")

jd = REPO / f"results/rules_runs/{a.run}/jobs"
existing = {f.name[:-len(".spec.json")].rsplit("_", 1)[-1]
            for f in jd.glob("*.spec.json")} if jd.exists() else set()
already = staged_pairs(jd)

plan = []                            # (jid, hash, ntask, payload)
for i, leg in enumerate(legs):
    if not leg:
        continue
    pl = (pd.concat([u[2] for u in leg], ignore_index=True)
          .sort_values(["task", "phrase"]).reset_index(drop=True))
    h = content_hash(pl)
    plan.append((f"{a.prefix}{i:02d}_{h}", h, len({u[1] for u in leg}), pl))

# overlap guard: a NEW leg (hash not on disk) whose pairs are already staged in
# some other leg means the worker rolls those (task,phrase) twice
old_pairs = {p for h, s in already.items() for p in s}
overlap = {}
for jid, h, _, pl in plan:
    if h in existing:
        continue
    hit = set(zip(pl.task, pl.phrase)) & old_pairs
    if hit:
        overlap[jid] = hit
if overlap:
    n_pairs = len(set().union(*overlap.values()))
    print(f"OVERLAP: {len(overlap)} new leg(s) repeat {n_pairs} (task,phrase) "
          f"pair(s) already staged in {jd} (payloads: "
          f"{sorted(already)[:6]}{'...' if len(already) > 6 else ''})")
    for jid, hit in list(overlap.items())[:3]:
        t, p = sorted(hit)[0]
        print(f"  {jid}: {len(hit)} pair(s), e.g. {t}: {p!r}")
    if not a.dry_run and not a.allow_overlap:
        raise SystemExit("refusing to stage duplicate rollouts; retire the stale "
                         "specs or pass --allow-overlap if a second roll is intended")
    print("WARNING: overlapping pairs would be rolled twice"
          + ("" if a.dry_run else " (--allow-overlap)"))

if not a.dry_run:
    ledger.to_parquet(D / "eval_dedup_ledger_v2.parquet", index=False)
    jd.mkdir(parents=True, exist_ok=True)

print(f"{'jid':<18} {'tasks':>5} {'phrases':>7} {'episodes':>8}  status")
n_new = n_skip = 0
for jid, h, ntask, pl in plan:
    if h in existing:
        status = "exists (skipped)"
        n_skip += 1
    elif a.dry_run:
        status = "dry-run"
    else:
        pl.to_parquet(jd / f"{jid}.payload.parquet", index=False)
        json.dump({"job_id": jid, "kind": "score", "method": "libero_bank_eval",
                   "prereg": a.prereg,
                   "rollout": {"inits": INITS, "port": 8000, "seed": 7}},
                  open(jd / f"{jid}.spec.json", "w"), indent=1)
        status = "written"
        n_new += 1
    print(f"{jid:<18} {ntask:>5} {len(pl):>7} {len(pl)*len(INITS):>8}  {status}")
if a.dry_run:
    print(f"dry-run: {len(plan)} legs planned for {jd}, nothing written")
else:
    print(f"staged {n_new} new jobs ({n_skip} already present) of {len(plan)} in {jd}")
