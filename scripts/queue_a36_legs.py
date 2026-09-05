#!/usr/bin/env python3
"""Queue A36 replicate legs: 6 books x 3 appliers x 3 conditions, one leg per (arm, task).

  FINAL_EVAL=1 .venv/bin/python scripts/queue_a36_legs.py [cond[,cond...]]

Default queues all three conditions; pass e.g. `nat` to stage only the natural
leg first (the A34-comparable one) and hold adv/orig for a later pass.

Fixed 2026-09-04: the previous regex `ph_a36_([tsb][23])_(adv|nat|orig)$` predated
commit 856b8c69, which gave the apply script an explicit applier and changed the
filenames to ph_a36_<tag>_<applier>_<cond>.parquet. It matched nothing. The job id
now carries the applier too, so arms stay separable from the id alone.
"""
import glob, hashlib, json, os, pathlib, re, sys
import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

CONDS = set((sys.argv[1] if len(sys.argv) > 1 else "adv,nat,orig").split(","))
R = pathlib.Path.home() / "dev/robotics/phrase-rl"
jd = R / "results/rules_runs/r1_sim/jobs"
jd.mkdir(parents=True, exist_ok=True)

RO = {"config": "config/experiment/simpler/pi0_finetune_bridge_ev.yaml",
      "ckpt": "juexzz/INTACT-pi0-finetune-rephrase-bridge", "seed": 42}
BASE = {"kind": "score", "method": "rollout", "draw": 0, "seed": 7,
        "proxy": {"C": 8.123, "bz": 0.4445, "bg": 11.3193}}
# matches A31/A34: natural is 24x1, adversarial and original are 24x2
REPS = {"adv": 2, "nat": 1, "orig": 2}
AP = {"claude": "cl", "gemini": "ge", "qwen": "qw"}

import subprocess, collections
_ls = subprocess.run(["git", "ls-tree", "-r", "--name-only", "origin/main", "--",
                      "results/rules_runs/r1_sim/jobs/"], capture_output=True, text=True).stdout
ORIGIN = {pathlib.Path(_l).name for _l in _ls.splitlines()}

tot = eps = 0
for f in sorted(glob.glob(str(R / "results/sealed/ph_a36_*.parquet"))):
    m = re.match(r"ph_a36_([tsb][23])_(claude|gemini|qwen)_(adv|nat|orig)$",
                 pathlib.Path(f).stem)
    if not m:
        continue
    tag, applier, cond = m.groups()
    if cond not in CONDS:
        continue
    rw = pd.read_parquet(f)
    n = rolled = 0
    for t, g in rw.groupby("task"):
        g = g[["task", "phrase"]].drop_duplicates()
        h = hashlib.sha1(pd.util.hash_pandas_object(g).values.tobytes()).hexdigest()[:10]
        jid = f"f36{tag}{AP[applier]}{cond[0]}_{h}"
        # jid is content-addressed: same payload -> same jid. Skip anything
        # already queued or finished ON ORIGIN, not just locally (2026-09-05 --
        # regenerated applies + a stripped local jobs dir caused duplicate legs).
        if (jd / f"{jid}.spec.json").exists() or f"{jid}.spec.json" in ORIGIN \
                or f"{jid}.result.parquet" in ORIGIN:
            continue
        g.to_parquet(jd / f"{jid}.payload.parquet", index=False)
        json.dump({"job_id": jid, **BASE,
                   "rollout": {**RO, "episode_ids": list(range(24)),
                               "repeats": REPS[cond]}},
                  open(jd / f"{jid}.spec.json", "w"), indent=1)
        n += 1
        rolled += len(g) * 24 * REPS[cond]
    tot += n
    eps += rolled
    print(f"f36{tag}{AP[applier]}{cond[0]}: {len(rw)} rows, {n} legs, {rolled} episodes")
print(f"A36 legs queued: {tot}  (~{eps:,} episodes)")
