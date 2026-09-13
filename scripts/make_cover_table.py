#!/usr/bin/env python3
"""results/analysis/cover_tab.tex — CoVer comparison table for the paper's
test-time-verification section. Rows = the three conditions CoVer was run on
(A37); columns = our arms (mean over the three appliers; rulebook = the
out-of-finetune diet, mean over draws) and CoVer under both metrics.
Everything final-state@60 except the last column (CoVer's native
first-success metric, 150-step horizon). The adversarial first-success cell
is computed here from the ever72_on episode records."""
import glob
import json
import pathlib

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
r1 = json.loads((R / "results/analysis/r1_cells.json").read_text())
rep = json.loads((R / "results/analysis/a36_cells.json").read_text())
org = json.loads((R / "results/analysis/orig_panel_cells.json").read_text())
cov = json.loads((R / "results/analysis/a37_cover_cells.json").read_text())

APS = ["claude", "gemini", "qwen"]


def s_cell(cond):
    """out-of-finetune diet, mean over draws then appliers."""
    vals = []
    for ap in APS:
        if cond == "orig":
            draws = [org[f"{b}|{ap}"]["pooled"] for b in ("s", "s2", "s3")]
        else:
            draws = [r1[f"r1s|{ap}|{cond}"]["pooled"]]
            for dr in ("2", "3"):
                rr = rep.get(f"s{dr}|{ap}|{cond}")
                if rr and rr.get("legs") == 12:
                    draws.append(rr["pooled"])
        vals.append(sum(draws) / len(draws))
    return sum(vals) / len(vals)


def sc_cell(cond):
    if cond == "orig":
        vals = [org[f"sc|{ap}"]["pooled"] for ap in APS]
    else:
        vals = [r1[f"sc|{ap}|{cond}"]["pooled"] for ap in APS]
    return sum(vals) / len(vals)


# adversarial CoVer first-success from the episode records
fs = glob.glob(str(R / "results/analysis/a37_cover_runs/ever72_on_k*.jsonl"))
E = pd.concat([pd.read_json(f, lines=True) for f in fs], ignore_index=True)
adv_ever = float(E.groupby(["task", "base"]).ever_success.mean()
                  .groupby("task").mean().mean() * 100)

NOREPH = {"adv": 24.5, "nat": r1["noreph|nat"]["pooled"], "orig": 36.1}
ROWS = [("adv", "Adversarial"), ("nat", "LLM-Generated Naturals"),
        ("orig", "Canonical")]
EVER = {"adv": adv_ever, "nat": cov["nat"]["on_ever"], "orig": cov["orig"]["on_ever"]}

lines = ["\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
         " & no & no-rules & out-of-finetune & CoVer & CoVer \\\\",
         " & rephraser & rephraser & rulebook & & (first-success) \\\\",
         "\\midrule"]
md = ["| condition | no rephraser | no-rules rephraser | out-of-finetune rulebook | CoVer | CoVer (first-success) |",
      "|---|---|---|---|---|---|"]
for cond, label in ROWS:
    v = [NOREPH[cond], sc_cell(cond), s_cell(cond), cov[cond]["on"], EVER[cond]]
    lines.append(f"{label} & " + " & ".join(f"{x:.1f}" for x in v) + " \\\\")
    md.append(f"| {label} | " + " | ".join(f"{x:.1f}" for x in v) + " |")
lines += ["\\bottomrule", "\\end{tabular}"]
out = R / "results/analysis/cover_tab.tex"
out.write_text("\n".join(lines) + "\n")
print("tex ->", out)
print()
print("\n".join(md))
print(f"\n(adv CoVer-off anchor for the prose: {cov['adv']['off']})")
