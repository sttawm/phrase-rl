#!/usr/bin/env python3
"""Queue A38 scaffold x Original jobs through the r1_sim queue (PREREG A38).

  FINAL_EVAL=1 .venv/bin/python scripts/queue_a38_scaffold.py apply   # qwen apply
  FINAL_EVAL=1 .venv/bin/python scripts/queue_a38_scaffold.py legs    # cl/ge rolls
  FINAL_EVAL=1 .venv/bin/python scripts/queue_a38_scaffold.py qwlegs  # qwen rolls

Same protocol as the A31 orig cells (24 layouts x 2 reps, seed 42, CRN):
spec shape copied from the b31*o legs. claude and gemini scaffold rewrites
are byte-identical (verified over three independent apply draws), so ONE
shared leg set (b38sco_*) serves both cells; rolling a second identical set
would produce byte-identical episodes under CRN. Qwen legs (b38scqwo_*)
are queued after its pod-side apply lands.
"""
import hashlib
import json
import os
import pathlib
import sys

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")
mode = sys.argv[1]
R = pathlib.Path.home() / "dev/robotics/phrase-rl"
jd = R / "results/rules_runs/r1_sim/jobs"
jd.mkdir(parents=True, exist_ok=True)

ROLLOUT_SPEC = {
    "kind": "score", "method": "rollout", "draw": 0, "seed": 7,
    "proxy": {"C": 8.123, "bz": 0.4445, "bg": 11.3193},
    "rollout": {"config": "config/experiment/simpler/pi0_finetune_bridge_ev.yaml",
                "ckpt": "juexzz/INTACT-pi0-finetune-rephrase-bridge",
                "seed": 42, "episode_ids": list(range(24)), "repeats": 2},
}


def rules_only(t):
    if "===RULES===" not in t:
        return t
    return "===RULES===" + t.split("===RULES===", 1)[1].split("===RATIONALE===", 1)[0]


def queue_legs(phrases, prefix):
    n = 0
    for task, grp in phrases.groupby("task"):
        h = hashlib.sha1((prefix + task + "\x00".join(sorted(grp.phrase))).encode()).hexdigest()[:10]
        jid = f"{prefix}_{h}"
        if (jd / f"{jid}.spec.json").exists():
            print("exists, skip", jid)
            continue
        grp[["task", "phrase"]].drop_duplicates().to_parquet(
            jd / f"{jid}.payload.parquet", index=False)
        json.dump({"job_id": jid, **ROLLOUT_SPEC},
                  open(jd / f"{jid}.spec.json", "w"), indent=1)
        print(f"queued {jid} ({task}, {grp.phrase.nunique()} phrase)")
        n += 1
    return n


if mode == "apply":
    rules = (R / "results/rules_runs/v2distill/scaffold.md").read_text()
    sha = hashlib.sha1(rules.encode()).hexdigest()
    orig12 = pd.read_parquet(R / "results/sealed/sealed_assets_gemini.parquet")
    bases = orig12.rename(columns={"nominal": "phrase"})[["task", "phrase"]].drop_duplicates()
    jid = f"a38scqwo_{sha[:8]}"
    if (jd / f"{jid}.spec.json").exists():
        raise SystemExit(f"exists: {jid}")
    bases.to_parquet(jd / f"{jid}.payload.parquet", index=False)
    json.dump({"job_id": jid, "kind": "apply",
               "rules_text": rules_only(rules), "rules_sha": sha[:12]},
              open(jd / f"{jid}.spec.json", "w"), indent=1)
    print("queued", jid, f"({len(bases)} bases)")
elif mode == "legs":
    ph = pd.read_parquet(R / "results/sealed/ph_a31_sc_claude_orig.parquet")
    ge = pd.read_parquet(R / "results/sealed/ph_a31_sc_gemini_orig.parquet")
    assert ph.set_index(["task", "base"]).phrase.equals(
        ge.set_index(["task", "base"]).phrase), "cl/ge rewrites diverged"
    print("legs queued:", queue_legs(ph[["task", "phrase"]], "b38sco"))
elif mode == "qwlegs":
    ph = pd.read_parquet(R / "results/sealed/ph_a31_sc_qwen_orig.parquet")
    print("qwen legs queued:", queue_legs(ph[["task", "phrase"]], "b38scqwo"))
