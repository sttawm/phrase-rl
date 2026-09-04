#!/usr/bin/env python3
"""Queue A36 qwen apply jobs: 6 replicate books x 3 sealed conditions.

  FINAL_EVAL=1 .venv/bin/python scripts/queue_a36_qwen.py

Qwen applies run pod-side (Qwen3.5-9B needs the GPU), unlike gemini/claude which
go through the API locally. Workers with /workspace/.can_apply pick these up.
Results land as <jid>.result.parquet with columns (task, phrase=BASE, rewrite);
scripts/collect_a36_qwen.py normalises them into the ph_a36_* schema.
"""
import hashlib, json, os, pathlib, sys
import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

# optional book filter: queue_a36_qwen.py s2      (default: all six)
ONLY = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
R = pathlib.Path.home() / "dev/robotics/phrase-rl"
jd = R / "results/rules_runs/r1_sim/jobs"
jd.mkdir(parents=True, exist_ok=True)


def rules_only(t):
    if "===RULES===" not in t:
        return t
    return "===RULES===" + t.split("===RULES===", 1)[1].split("===RATIONALE===", 1)[0]


orig12 = pd.read_parquet(R / "results/sealed/sealed_assets_gemini.parquet")
BASES = {
    "adv": pd.concat([
        orig12.rename(columns={"ert_instruction": "phrase"})[["task", "phrase"]],
        pd.read_parquet(R / "results/sealed/a29_new_erts.parquet")[["task", "phrase"]],
    ]).drop_duplicates().reset_index(drop=True),
    "nat": pd.read_parquet(R / "results/sealed/ph_sealed_natural_v2_img.parquet")[["task", "phrase"]],
    "orig": orig12.rename(columns={"nominal": "phrase"})[["task", "phrase"]].drop_duplicates(),
}

n = 0
for book, tag in (("train_only_r2", "t2"), ("train_only_r3", "t3"),
                  ("sim_only_r2", "s2"), ("sim_only_r3", "s3"),
                  ("both_r2", "b2"), ("both_r3", "b3")):
    if ONLY and tag not in ONLY:
        continue
    rules = (R / "results/rules_runs/v2distill" / book / "rules.md").read_text()
    sha = hashlib.sha1(rules.encode()).hexdigest()
    for cond, bases in BASES.items():
        jid = f"g36{tag}qw{cond[0]}_{sha[:8]}"
        if (jd / f"{jid}.spec.json").exists():
            print("exists, skip", jid)
            continue
        bases[["task", "phrase"]].drop_duplicates().to_parquet(
            jd / f"{jid}.payload.parquet", index=False)
        json.dump({"job_id": jid, "kind": "apply",
                   "rules_text": rules_only(rules), "rules_sha": sha[:12]},
                  open(jd / f"{jid}.spec.json", "w"), indent=1)
        print(f"queued {jid}  ({cond}, {len(bases.drop_duplicates())} bases)")
        n += 1
print("A36 qwen apply jobs queued:", n)
