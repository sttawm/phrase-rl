#!/usr/bin/env python3
"""Queue A34 qwen apply jobs: 4 books x the 192 image-conditioned naturals."""
import hashlib, json, os, pathlib, sys
import pandas as pd
if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")
R = pathlib.Path.home() / "dev/robotics/phrase-rl"
jd = R / "results/rules_runs/r1_sim/jobs"
nat = pd.read_parquet(R / "results/sealed/ph_sealed_natural_v2_img.parquet")[["task", "phrase"]]
def rules_only(t):
    return "===RULES===" + t.split("===RULES===", 1)[1].split("===RATIONALE===", 1)[0] \
        if "===RULES===" in t else t
for tag, path in (("t", "v2distill/train_only/rules.md"), ("s", "v2distill/sim_only/rules.md"),
                  ("b", "v2distill/both/rules.md"), ("sc", "v2distill/scaffold.md")):
    rules = (R / "results/rules_runs" / path).read_text()
    jid = f"a00qw34{tag}_" + hashlib.sha1(rules.encode()).hexdigest()[:8]
    nat.drop_duplicates().to_parquet(jd / f"{jid}.payload.parquet", index=False)
    json.dump({"job_id": jid, "kind": "apply", "rules_text": rules_only(rules),
               "rules_sha": hashlib.sha1(rules.encode()).hexdigest()[:12]},
              open(jd / f"{jid}.spec.json", "w"), indent=1)
    print("queued", jid, len(nat))
