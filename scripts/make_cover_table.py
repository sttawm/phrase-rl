#!/usr/bin/env python3
"""results/analysis/cover_tab.tex — 3x3 baselines table: CoVer vs the
no-rephraser and no-rules rephraser baselines on the three conditions CoVer
was run on (A37). All cells final-state@60; no-rules rephraser = mean over
the three appliers."""
import json
import pathlib

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


NOREPH = {"adv": 24.5, "nat": r1["noreph|nat"]["pooled"], "orig": 36.1}
ROWS = [("adv", "Adversarial"), ("nat", "LLM-Generated Naturals"),
        ("orig", "Canonical")]

lines = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule",
         " & no rephraser & no-rules rephraser & CoVer \\\\",
         "\\midrule"]
md = ["| condition | no rephraser | no-rules rephraser | CoVer |",
      "|---|---|---|---|"]
for cond, label in ROWS:
    v = [NOREPH[cond], sc_cell(cond), cov[cond]["on"]]
    lines.append(f"{label} & " + " & ".join(f"{x:.1f}" for x in v) + " \\\\")
    md.append(f"| {label} | " + " | ".join(f"{x:.1f}" for x in v) + " |")
lines += ["\\bottomrule", "\\end{tabular}"]
out = R / "results/analysis/cover_tab.tex"
out.write_text("\n".join(lines) + "\n")
print("tex ->", out)
print()
print("\n".join(md))
print(f"\n(adv CoVer-off anchor for the prose: {cov['adv']['off']})")
