#!/usr/bin/env python3
"""LaTeX p-score grids for the paper — out-of-finetune rulebook only.

Reads results/analysis/dashboard_pscores.json (written by
pscores_dashboard.py) and emits two plain tabular fragments:

  results/analysis/pscores_tab_baseline.tex   vs the no-rephraser baseline
  results/analysis/pscores_tab_scaffold.tex   vs the same applier's no-rules
                                              rephraser scaffold

Each cell: relative change in success (percent of the baseline rate; the
three rulebook draws pooled before pairing) and the paired-by-base sign-flip
permutation p. Copied into the paper repo as figures/pscores_tab_*.tex and
\\input there (booktabs rules).
"""
import json
import pathlib

R = pathlib.Path(__file__).resolve().parents[1]
J = json.loads((R / "results/analysis/dashboard_pscores.json").read_text())

CONDS = ["adv", "hum", "nat"]
HDR = (" & Adversarial & Human-Generated & LLM-Generated \\\\\n"
       " & (n=72) & Naturals (n=392) & Naturals (n=186) \\\\\n")
APS = [("claude", "Claude"), ("gemini", "Gemini"), ("qwen", "Qwen")]


def cell(rec):
    p = rec["p_perm"]
    ps = "p<0.0001" if p < 1e-4 else f"p={p:.4f}"
    return f"{rec['pct']:+.1f}\\%~({ps})"


def emit(suffix, fname):
    lines = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule",
             HDR.rstrip(), "\\midrule"]
    for ap, ap_lbl in APS:
        row = [cell(J[f"{c}|{ap}|s{suffix}"]) for c in CONDS]
        lines.append(f"{ap_lbl} & " + " & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out = R / f"results/analysis/{fname}"
    out.write_text("\n".join(lines) + "\n")
    print("tex ->", out)


emit("", "pscores_tab_baseline.tex")
emit("_vs_sc", "pscores_tab_scaffold.tex")

# provenance printout: the pooled rates behind each percentage
for suffix in ["", "_vs_sc"]:
    for c in CONDS:
        for ap, _ in APS:
            r = J[f"{c}|{ap}|s{suffix}"]
            print(f"{c}|{ap}|s{suffix or ''}: cell {r['cell_mean']} vs base "
                  f"{r['base_mean']} -> {r['pct']:+.1f}%  p={r['p_perm']}")
