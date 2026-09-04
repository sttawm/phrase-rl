#!/usr/bin/env python3
"""Queue A36 replicate legs: 6 books x 3 conditions, one leg per (arm, task)."""
import glob, hashlib, json, os, pathlib, re
import pandas as pd
if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")
R = pathlib.Path.home() / "dev/robotics/phrase-rl"
jd = R / "results/rules_runs/r1_sim/jobs"
RO = {"config": "config/experiment/simpler/pi0_finetune_bridge_ev.yaml",
      "ckpt": "juexzz/INTACT-pi0-finetune-rephrase-bridge", "seed": 42}
BASE = {"kind": "score", "method": "rollout", "draw": 0, "seed": 7,
        "proxy": {"C": 8.123, "bz": 0.4445, "bg": 11.3193}}
REPS = {"adv": 2, "nat": 1, "orig": 2}
tot = 0
for f in sorted(glob.glob(str(R / "results/sealed/ph_a36_*.parquet"))):
    m = re.match(r"ph_a36_([tsb][23])_(adv|nat|orig)$", pathlib.Path(f).stem)
    if not m:
        continue
    tag, cond = m.groups()
    rw = pd.read_parquet(f)
    n = 0
    for t, g in rw.groupby("task"):
        g = g[["task", "phrase"]].drop_duplicates()
        h = hashlib.sha1(pd.util.hash_pandas_object(g).values.tobytes()).hexdigest()[:10]
        jid = f"f36{tag}{cond[0]}_{h}"
        if (jd / f"{jid}.spec.json").exists():
            continue
        g.to_parquet(jd / f"{jid}.payload.parquet", index=False)
        json.dump({"job_id": jid, **BASE,
                   "rollout": {**RO, "episode_ids": list(range(24)), "repeats": REPS[cond]}},
                  open(jd / f"{jid}.spec.json", "w"), indent=1)
        n += 1
    tot += n
    print(f"f36{tag}{cond[0]}: {len(rw)} phrases, {n} legs")
print("A36 legs queued:", tot)
