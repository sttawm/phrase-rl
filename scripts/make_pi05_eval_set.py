#!/usr/bin/env python3
"""Assemble the pi0.5 FINAL_EVAL task set from the canonical screens.

Strata (PREREG 2026-09-04): A in-finetune >=90% | B OOD >=80% | C OOD 5% |
D OOD 0% negative control (seeded sample). Writes eval_set.json:
  {"tasks": {"suite:tid": canonical}, "strata": {...}, "screen": {...}}

  .venv/bin/python scripts/make_pi05_eval_set.py [--n-infinetune 12] [--n-floor 5]
"""
import argparse, glob, json, pathlib, random, collections
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
D = REPO / "results/analysis/pi05_bank"
ap = argparse.ArgumentParser()
ap.add_argument("--n-infinetune", type=int, default=12)
ap.add_argument("--n-floor", type=int, default=5)
ap.add_argument("--seed", type=int, default=20260904)
a = ap.parse_args()

# --- screens: OOD from committed jsonl; in-finetune from the reserve results ---
def agg(rows):
    c = collections.defaultdict(lambda: [0, 0, ""])
    for r in rows:
        k = f'{r["suite"]}:{r["task_id"]}'
        c[k][0] += r["success"]; c[k][1] += 1; c[k][2] = r.get("canonical", "")
    return {k: (s, n, cn) for k, (s, n, cn) in c.items()}

ood_rows = []
for f in glob.glob(str(D / "screens/*.jsonl")):
    ood_rows += [json.loads(l) for l in open(f) if l.strip()]
ood = agg(ood_rows)

res = {}
for jid in ("reserve_screen_canon_w0", "reserve_screen2_canon_w0"):
    p = REPO / f"results/rules_runs/p_seal/jobs/{jid}.result.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        for r in d.itertuples():
            res[str(r.task)] = (float(r.gt_success), int(r.n_ctx), str(r.phrase))

splits = json.load(open(D / "splits.json"))
lang = {f'{t["suite"]}:{t["task_id"]}': t["lang"]
        for t in splits["sealed_test"] + splits["val"]}
origin = {f'{t["suite"]}:{t["task_id"]}': "sealed" for t in splits["sealed_test"]}
origin.update({f'{t["suite"]}:{t["task_id"]}': "val" for t in splits["val"]})

rng = random.Random(a.seed)
A = sorted([k for k, (pct, n, _) in res.items() if pct >= 90], key=str)[:a.n_infinetune]
B = sorted([k for k, (s, n, _) in ood.items() if n >= 20 and 100*s/n >= 80])
C = sorted([k for k, (s, n, _) in ood.items() if n >= 20 and 0 < 100*s/n < 80])
Dz = sorted([k for k, (s, n, _) in ood.items() if n >= 20 and s == 0])
rng.shuffle(Dz); Dz = sorted(Dz[:a.n_floor])

tasks, strata, screen = {}, {}, {}
for k in A:
    tasks[k] = res[k][2]; strata[k] = "A_in_finetune"; screen[k] = res[k][0]
for k, lab in [(x, "B_ood_range") for x in B] + [(x, "C_ood_marginal") for x in C] \
             + [(x, "D_ood_floor") for x in Dz]:
    s, n, cn = ood[k]
    tasks[k] = cn or lang.get(k, ""); strata[k] = lab; screen[k] = round(100*s/n, 1)

out = {"prereg": "2026-09-04 FINAL_EVAL design", "seed": a.seed,
       "tasks": tasks, "strata": strata, "screen_pct": screen,
       "origin": {k: origin.get(k, "reserve") for k in tasks}}
(D / "eval_set.json").write_text(json.dumps(out, indent=1))
n_by = collections.Counter(strata.values())
print(f"eval_set.json: {len(tasks)} tasks  {dict(n_by)}")
for k in tasks:
    print(f"  [{strata[k][0]}] {k:22s} {screen[k]:5.1f}%  {tasks[k][:52]}")
