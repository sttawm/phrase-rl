#!/usr/bin/env python3
"""Aggregate club-ranking features into the global training-phrase ranking.

Reads results/analysis/club_rank_features.parquet (one row per instruction x
episode x frame from fine_exam_score.py, where task==phrase==instruction).
Per instruction: mean grip_row (primary, lower error = better phrase) and mean
z_row over its ~C x 4 evals; global rank by grip. eff_C per instruction recorded
(>=20 -> ~92% coarse sign-accuracy cell; 16-19 -> ~89%).
Caveat recorded in output: scores are per-instruction own-scene estimates (the
deployment estimator), so cross-instruction comparisons carry scene-difficulty
variance in addition to phrase quality.
"""
import json

import pandas as pd

d = pd.read_parquet("results/analysis/club_rank_features.parquet")
meta = pd.read_parquet("results/analysis/club_rank_phrases.parquet").set_index("task")
agg = d.groupby("task").agg(grip=("grip_row", "mean"), z=("z_row", "mean"),
                            n_evals=("z_row", "size")).reset_index()
agg = agg.rename(columns={"task": "instruction"})
agg["eff_C"] = agg.instruction.map(meta.eff_C)
agg["n_club_eps"] = agg.instruction.map(meta.n_club_eps)
agg = agg.sort_values("grip").reset_index(drop=True)
agg["rank_grip"] = range(1, len(agg) + 1)
agg.to_parquet("results/analysis/club_phrase_ranking.parquet", index=False)
top = agg.head(15)[["rank_grip", "instruction", "grip", "z", "eff_C"]].to_dict("records")
bot = agg.tail(10)[["rank_grip", "instruction", "grip", "z", "eff_C"]].to_dict("records")
json.dump({"estimator": "F=4 x C<=20 own-club contexts (grip primary)",
           "caveat": "own-scene estimates: cross-instruction comparison includes scene difficulty",
           "top15": top, "bottom10": bot},
          open("results/analysis/club_ranking_summary.json", "w"), indent=1)
print(agg.head(10)[["rank_grip", "instruction", "grip", "eff_C"]].to_string(index=False))
print(f"-> {len(agg)} instructions ranked; club_phrase_ranking.parquet + club_ranking_summary.json")
