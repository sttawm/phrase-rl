#!/usr/bin/env python3
"""results/analysis/draws_tab.tex — per-draw rulebook grid for the paper
appendix (the by-draw dashboard row, as a table). Success %, pooled over the
three appliers, per evidence diet x independent distillation draw x
condition. Data: r1_cells.json / a36_cells.json / a39_human_cells.json,
same access logic as make_rules_dashboard_compact.py."""
import json
import pathlib

R = pathlib.Path(__file__).resolve().parents[1]
r1 = json.loads((R / "results/analysis/r1_cells.json").read_text())
rep = json.loads((R / "results/analysis/a36_cells.json").read_text())
hum = json.loads((R / "results/analysis/a39_human_cells.json").read_text())

APS = ["claude", "gemini", "qwen"]
DIETS = [("s", "out-of-finetune"), ("b", "both"), ("t", "in-finetune")]
CONDS = ["adv", "hum", "nat"]
HUM_BOOK = {("s", 1): "s", ("s", 2): "s2", ("s", 3): "s3",
            ("b", 1): "b", ("b", 2): "b2", ("b", 3): "b3",
            ("t", 1): "t", ("t", 2): "t2", ("t", 3): "t3"}


def rec(cond, diet, ap, dr):
    if cond == "hum":
        return hum.get(f"{HUM_BOOK[(diet, dr)]}|{ap}")
    if dr == 1:
        return r1.get(f"r1{diet}|{ap}|{cond}")
    r = rep.get(f"{diet}{dr}|{ap}|{cond}")
    return r if (r and r.get("legs") == 12) else None


def cell(cond, diet, dr):
    vals = [rec(cond, diet, ap, dr)["pooled"] for ap in APS
            if rec(cond, diet, ap, dr)]
    return sum(vals) / len(vals)


lines = ["\\begin{tabular}{@{}llccc@{}}", "\\toprule",
         " & & Adversarial & Human-Generated & LLM-Generated \\\\",
         " & & & Naturals & Naturals \\\\"]
for diet, dlabel in DIETS:
    lines.append("\\midrule")
    for dr in (1, 2, 3):
        lead = dlabel if dr == 1 else ""
        row = " & ".join(f"{cell(c, diet, dr):.1f}" for c in CONDS)
        lines.append(f"{lead} & rulebook-{dr} & {row} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
out = R / "results/analysis/draws_tab.tex"
out.write_text("\n".join(lines) + "\n")
print("tex ->", out)
print("\n".join(lines))
