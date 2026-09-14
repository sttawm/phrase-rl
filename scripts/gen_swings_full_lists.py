#!/usr/bin/env python3
"""paper/swings_full_{bridge,libero}_body.tex — the FULL set of significant
swing sets (swing_sets_*.csv) in the paper's list idiom: category headers,
task + gap + exact p per set, \\pct rungs with token-diff highlights (best =
hi, all others = lo, matching the curated fragments). Input to the companion
doc paper/swings_full.tex shipped in the artifacts repo."""
import math
import pathlib
import re

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
COLORS = {"purple", "black", "yellow", "red", "blue", "green", "white",
          "grey", "gray", "orange", "pink", "brown", "silver"}

BRIDGE_ORDER = ["Source noun synonym", "Destination noun synonym",
                "Preposition", "Color --- addition helps",
                "Color --- addition hurts", "Action verb",
                "Phrase structure", "Multi-edit"]
LIBERO_ORDER = ["Noun synonym", "Color", "Action verb", "Preposition"]


def esc(s):
    return (s.replace("\\", "").replace("&", "\\&").replace("%", "\\%")
             .replace("#", "\\#").replace("_", "\\_"))


def toks(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def header(g, cat):
    if cat == "noun":
        return "Noun synonym"
    if cat == "verb":
        return "Action verb"
    if cat == "preposition":
        return "Preposition"
    if cat == "case/structure":
        return "Phrase structure"
    if cat == "multi-edit":
        return "Multi-edit"
    if cat in ("source noun", "destination noun"):
        return cat.capitalize().replace("noun", "noun synonym")
    # color/modifier: helps if the best phrase carries the color word
    best = g[g.is_best].phrase.iloc[0]
    worst = g.loc[g.succ_pct.idxmin(), "phrase"]
    helps = COLORS & (set(toks(best)) - set(toks(worst)))
    return f"Color --- addition {'helps' if helps else 'hurts'}"


def hilite(phrase, common, box):
    out = []
    for w in phrase.split():
        if re.sub(r"[^a-z0-9']", "", w.lower()) in common:
            out.append(esc(w))
        else:
            out.append("\\colorbox{%s}{\\strut %s}" % (box, esc(w)))
    return " ".join(out)


def set_p(g):
    if "min_p" in g.columns and g.min_p.notna().any():
        p = g.min_p.iloc[0]
    else:
        p = math.erfc(g.max_z.iloc[0] / math.sqrt(2))
    return "p<0.0001" if p < 1e-4 else f"p={p:.4f}"


def emit(df, order, libero, out):
    df = df.copy()
    df["hdr"] = [header(df[df.set_id == s], c)
                 for s, c in zip(df.set_id, df.category)]
    lines = []
    for hdr in order:
        sub = df[df.hdr == hdr]
        if sub.empty:
            continue
        lines.append("{\\bfseries %s}\\par\\nopagebreak" % hdr)
        sets = (sub.groupby("set_id")
                   .agg(gap=("set_gap_pp", "first"), task=("task", "first"))
                   .sort_values("gap", ascending=False))
        for sid, srow in sets.iterrows():
            g = df[df.set_id == sid].sort_values("succ_pct", ascending=False)
            common = set.intersection(*(set(toks(p)) for p in g.phrase))
            task = esc(srow.task.replace("widowx_", "").replace("_clean", ""))
            lines.append("{\\leftskip=0.7em")
            lines.append(
                "{\\bfseries %s}\\enspace +%d\\,pp\\,$\\cdot$\\, %s"
                "\\par\\nopagebreak" % (task, round(srow.gap), set_p(g)))
            for _, r in g.iterrows():
                col, box = ("hipct", "hi") if r.is_best else ("lopct", "lo")
                lines.append(
                    "\\hangindent=2.5em \\pct{%s}{%d}~\\texttt{%s}\\par"
                    % (col, round(r.succ_pct), hilite(r.phrase, common, box)))
            lines.append("}")
            lines.append("\\vspace{3.2pt}")
    out.write_text("\n".join(lines) + "\n")
    print(f"{out.name}: {df.set_id.nunique()} sets, {len(df)} rungs")


B = pd.read_csv(R / "results/analysis/swing_sets_bridge.csv")
L = pd.read_csv(R / "results/analysis/swing_sets_libero.csv")
emit(B, BRIDGE_ORDER, False, R / "paper/swings_full_bridge_body.tex")
emit(L, ["Noun synonym", "Color --- addition helps",
         "Color --- addition hurts", "Action verb", "Preposition"], True,
     R / "paper/swings_full_libero_body.tex")
