#!/usr/bin/env python3
"""Scene-ambiguity pass on the LIBERO swing sets (user ruling 2026-09-15):
a set is removed when a generalized noun legally matches more than one
object in the scene, so its gap measures scene resolution, not phrasing.
Filters results/analysis/swing_sets_libero.csv in place.

Removed (8 sets):
  libero_object/4|noun/other   red bottle / sauce could describe other objects
  libero_object/9|noun/other   "pick up the bottle" -- a second bottle in scene
  libero_object/1|noun/other   box / package / container resolve to another object
  libero_object/3|noun/other   more than one bottle in scene
  libero_object/8|noun/other   more than one bottle in scene
  libero_object/6|noun/other   "block" resolves to another object
  libero_object/4|color        "bottle" resolves to another object
  libero_goal/9|noun/other     "stand" could be the drawers too; both non-best
                               rungs are stand-phrases, so the set collapses
"""
import pathlib

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
REMOVED = [
    "libero_object/4|noun/other",
    "libero_object/9|noun/other",
    "libero_object/1|noun/other",
    "libero_object/3|noun/other",
    "libero_object/8|noun/other",
    "libero_object/6|noun/other",
    "libero_object/4|color",
    "libero_goal/9|noun/other",
]

path = R / "results/analysis/swing_sets_libero.csv"
L = pd.read_csv(path)
missing = [s for s in REMOVED if s not in set(L.set_id)]
assert not missing, missing
out = L[~L.set_id.isin(REMOVED)]
out.to_csv(path, index=False)
print(f"{path.name}: {L.set_id.nunique()} -> {out.set_id.nunique()} sets "
      f"({len(L)} -> {len(out)} phrase rows), {out.task.nunique()} tasks")
