#!/usr/bin/env python3
"""Aggregate the club-phrase ranking features into per-task ranked tables.

Reads results/analysis/club_rank_features.parquet (one row per task/phrase/ep/t
from fine_exam_score.py). Aggregation mirrors the measured-grid estimator:
per phrase, mean grip_row and mean z_row over its (episode x frame) evals, then
rank within task (rank-01 space) and blend c4b = 0.25*z01 + 0.75*g01.
Confidence note: effective C recorded per task; >=20 -> ~92% coarse sign acc
(grip), 16-19 -> ~89%.
"""
import json

import numpy as np
import pandas as pd

d = pd.read_parquet("results/analysis/club_rank_features.parquet")
rows = []
for task, sub in d.groupby("task"):
    agg = sub.groupby("phrase").agg(z=("z_row", "mean"), grip=("grip_row", "mean"),
                                    n_evals=("z_row", "size")).reset_index()
    eff_c = sub.groupby("phrase").episode_index.nunique().min()
    r01 = lambda x: pd.Series(np.argsort(np.argsort(x)) / max(len(x) - 1, 1), index=x.index)
    agg["g01"] = r01(-agg.grip)
    agg["z01"] = r01(agg.z)
    agg["c4b"] = 0.25 * agg.z01 + 0.75 * agg.g01
    agg = agg.sort_values("g01", ascending=False)
    agg["rank_grip"] = range(1, len(agg) + 1)
    agg["task"] = task
    agg["eff_C"] = eff_c
    rows.append(agg)
out = pd.concat(rows, ignore_index=True)
out.to_parquet("results/analysis/club_phrase_ranking.parquet", index=False)
top = {t: s.head(3)[["phrase", "grip", "c4b"]].to_dict("records")
       for t, s in out.groupby("task")}
json.dump(top, open("results/analysis/club_ranking_top3.json", "w"), indent=1)
print(out.groupby("task").agg(phrases=("phrase", "size"), eff_C=("eff_C", "first")))
print("-> results/analysis/club_phrase_ranking.parquet + club_ranking_top3.json")
