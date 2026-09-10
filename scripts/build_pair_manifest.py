#!/usr/bin/env python3
"""Merge the two minimal-pair candidate sets into one roll manifest.

Set A: the top-20 curated pairs from minimal_pairs_top20.md (exact strings,
mined from pooled r1_sim legs) plus a color-word-insertion extension mined for
the color family (results/analysis/color_add_pairs.csv, selected below).
Set B: results/analysis/bridge_pair_candidates.parquet (32 phrases, other
session, commit 3d5bba6c) — kept whole, groups preserved.

Dedupe on (task, exact phrase string) ONLY; provenance recorded per row (a
deduped row keeps every source, joined by '+'). Booleans per group:
single_concept (exactly one thing differs within the group's pairs) and
format_matched (same capitalisation + trailing punctuation across the group).
Phrases are rolled EXACTLY as written — never normalise case or punctuation
(measured here: trailing full stop -11.1pp, initial capital -11.7pp).

Emits results/analysis/bridge_pair_manifest.parquet.
"""
import pathlib

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]

# (group, single_concept, format_matched, [(label, task, phrase, prior_succ, prior_n)])
A = [
 ("A01_case_period", False, False, [
   ("lo cased+period", "widowx_pepsi_on_plate_clean", "Set the Pepsi down on the plate.", 19.4, 432),
   ("hi lowercase", "widowx_pepsi_on_plate_clean", "set the pepsi down on the plate", 82.3, 96)]),
 ("A02_brand_case", True, False, [
   ("lo Pepsi cased", "widowx_pepsi_on_plate_clean", "Put the Pepsi on the plate.", 18.3, 240),
   ("hi lowercase", "widowx_pepsi_on_plate_clean", "put the pepsi on the plate", 61.5, 96)]),
 ("A03_prep_onto_on", True, True, [
   ("lo onto", "widowx_stack_cube", "place the green block onto the yellow block", 8.3, 24),
   ("hi on", "widowx_stack_cube", "place the green block on the yellow block", 54.2, 24)]),
 ("A04_prep_on_atop", True, True, [
   ("lo on", "widowx_spoon_on_towel", "put the spoon on the towel", 30.4, 69),
   ("hi atop", "widowx_spoon_on_towel", "put the spoon atop the towel", 66.7, 72)]),
 ("A05_prep_on_into", True, True, [
   ("lo on", "widowx_coke_can_on_wheel_clean", "Put the red can on the black wheel.", 4.2, 24),
   ("hi into", "widowx_coke_can_on_wheel_clean", "Put the red can into the black wheel.", 31.2, 96)]),
 ("A06_prep_in_inside", True, True, [
   ("lo in", "widowx_carrot_on_ramekin_clean", "Place the carrot in the ramekin.", 19.3, 192),
   ("hi inside", "widowx_carrot_on_ramekin_clean", "Place the carrot inside the ramekin.", 40.6, 96)]),
 ("A07_prep_to_onto", True, True, [
   ("lo to", "widowx_pepsi_on_plate_clean", "Move the blue can to the plate.", 16.7, 48),
   ("hi onto", "widowx_pepsi_on_plate_clean", "Move the blue can onto the plate.", 37.5, 168)]),
 ("A08_verb_position_put", True, True, [
   ("lo position", "widowx_nut_on_wheel_clean", "Position the nut right on the wheel.", 4.2, 120),
   ("hi put", "widowx_nut_on_wheel_clean", "Put the nut right on the wheel.", 45.8, 48)]),
 ("A09_verb_put_place", True, True, [
   ("lo put", "widowx_cube_on_plate_clean", "put the green cube into the dish", 37.5, 24),
   ("hi place", "widowx_cube_on_plate_clean", "place the green cube into the dish", 75.0, 48)]),
 ("A10_verb_place_deposit", True, True, [
   ("lo place", "widowx_eggplant_on_sponge_clean", "Place the purple eggplant onto the yellow and green sponge.", 60.4, 48),
   ("hi deposit", "widowx_eggplant_on_sponge_clean", "Deposit the purple eggplant onto the yellow and green sponge.", 91.7, 24)]),
 ("A11_verb_move_set", True, True, [
   ("lo move", "widowx_eggplant_on_sponge_clean", "Move the purple eggplant onto the sponge.", 59.5, 168),
   ("hi set", "widowx_eggplant_on_sponge_clean", "Set the purple eggplant onto the sponge.", 89.6, 48)]),
 ("A12_verb_move_rest", True, True, [
   ("lo move", "widowx_eggplant_on_keyboard_clean", "Move the purple eggplant on the black keyboard.", 16.7, 24),
   ("hi rest", "widowx_eggplant_on_keyboard_clean", "Rest the purple eggplant on the black keyboard.", 45.8, 24)]),
 ("A13_noun_thing_eggplant", True, True, [
   ("lo thing", "widowx_eggplant_on_sponge_clean", "set the purple thing on the yellow and green sponge", 22.9, 96),
   ("hi eggplant", "widowx_eggplant_on_sponge_clean", "set the purple eggplant on the yellow and green sponge", 72.9, 96)]),
 ("A14_noun_soda_pepsi", True, True, [
   ("lo soda", "widowx_pepsi_on_plate_clean", "set the soda on the plate", 29.2, 24),
   ("hi pepsi", "widowx_pepsi_on_plate_clean", "set the pepsi on the plate", 75.0, 48)]),
 ("A15_noun_can_cylinder", True, True, [
   ("lo can", "widowx_coke_can_on_keyboard_clean", "Put the red soda can on top of the black keyboard.", 2.8, 72),
   ("hi cylinder", "widowx_coke_can_on_keyboard_clean", "Put the red soda cylinder on top of the black keyboard.", 45.8, 48)]),
 ("A16_noun_vegetable_carrot", True, True, [
   ("lo vegetable", "widowx_carrot_on_ramekin_clean", "Pick up the orange vegetable and place it in the white bowl.", 12.5, 24),
   ("hi carrot", "widowx_carrot_on_ramekin_clean", "Pick up the orange carrot and place it in the white bowl.", 54.2, 48)]),
 ("A17_noun_dish_plate", True, True, [
   ("lo dish", "widowx_coke_can_on_plate_clean", "put the coke can on the dish", 0.0, 60),
   ("hi plate", "widowx_coke_can_on_plate_clean", "put the coke can on the plate", 38.1, 105)]),
 ("A18_order_fronted", True, True, [
   ("lo fronted", "widowx_eggplant_on_sponge_clean", "purple Eggplant set on the sponge.", 45.8, 24),
   ("hi canonical order", "widowx_eggplant_on_sponge_clean", "Set the purple eggplant on the sponge.", 83.3, 48)]),
 ("A19_articles", True, True, [
   ("lo with articles", "widowx_eggplant_on_keyboard_clean", "put the eggplant on the keyboard", 12.0, 192),
   ("hi bare canonical", "widowx_eggplant_on_keyboard_clean", "put eggplant on keyboard", 37.5, 672)]),
 ("A20_polite_colororder", False, True, [
   ("lo bare", "widowx_eggplant_on_sponge_clean", "put the purple eggplant on the yellow and green sponge", 62.1, 240),
   ("hi please+swap", "widowx_eggplant_on_sponge_clean", "please put the purple eggplant on the green and yellow sponge", 83.3, 24)]),
 # --- color-word insertion extension (both signs, object vs destination) ---
 ("Ac1_dest_black_keyboard", True, True, [
   ("bare", "widowx_eggplant_on_keyboard_clean", "put the eggplant on the keyboard", 12.0, 192),
   ("colored", "widowx_eggplant_on_keyboard_clean", "put the eggplant on the black keyboard", 54.2, 96)]),
 ("Ac2_dest_black_wheel", True, False, [
   ("bare", "widowx_coke_can_on_wheel_clean", "grab the red coke can and set it on the wheel", 8.3, 72),
   ("colored", "widowx_coke_can_on_wheel_clean", "Grab the red Coke can and set it on the black wheel.", 38.5, 96)]),
 ("Ac3_dest_yellow_plate", True, False, [
   ("bare", "widowx_cube_on_plate_clean", "Pick up the green cube and put it on the plate.", 89.6, 48),
   ("colored", "widowx_cube_on_plate_clean", "pick up the green cube and put it on the yellow plate", 41.7, 24)]),
 ("Ac4_dest_green_plate_collision", True, True, [
   ("bare", "widowx_cube_on_plate_clean", "Place the green block on the plate.", 86.7, 120),
   ("colored", "widowx_cube_on_plate_clean", "Place the green block on the green plate.", 50.0, 24)]),
 ("Ac5_obj_purple_eggplant", True, True, [
   ("bare", "widowx_eggplant_on_sponge_clean", "eggplant goes on the sponge", 4.2, 24),
   ("colored", "widowx_eggplant_on_sponge_clean", "purple eggplant goes on the sponge", 66.7, 24)]),
 ("Ac6_obj_green_block", True, True, [
   ("bare", "widowx_cube_on_plate_clean", "put the block on the plate", 52.1, 48),
   ("colored", "widowx_cube_on_plate_clean", "put the green block on the plate", 87.5, 120)]),
 ("Ac7_obj_blue_coke", True, True, [
   ("bare", "widowx_pepsi_on_plate_clean", "set the coke on the plate", 22.9, 48),
   ("colored", "widowx_pepsi_on_plate_clean", "set the blue coke on the plate", 70.8, 48)]),
 ("Ac8_obj_grey_nut", True, True, [
   ("bare", "widowx_nut_on_plate_clean", "drop the nut on the yellow plate", 45.8, 24),
   ("colored", "widowx_nut_on_plate_clean", "drop the grey nut on the yellow plate", 4.2, 24)]),
 ("Ac9_obj_purple_on_colored_dest", True, True, [
   ("bare", "widowx_eggplant_on_keyboard_clean", "The eggplant needs to end up on the black keyboard.", 58.3, 24),
   ("colored", "widowx_eggplant_on_keyboard_clean", "The purple eggplant needs to end up on the black keyboard.", 20.8, 48)]),
 ("Ac10_obj_orange_carrot", True, False, [
   ("bare", "widowx_carrot_on_plate", "put the carrot on the plate", 25.0, 84),
   ("colored", "widowx_carrot_on_plate", "Put the orange carrot on the plate.", 62.5, 24)]),
]

rows = []
for group, single, fmt, members in A:
    src = "A_color" if group.startswith("Ac") else "A_top20"
    for label, task, phrase, ps, pn in members:
        rows.append({"task": task, "group": group, "label": label, "phrase": phrase,
                     "prior_success": ps, "prior_n": pn, "source": src,
                     "single_concept": single, "format_matched": fmt})

B = pd.read_parquet(R / "results/analysis/bridge_pair_candidates.parquet")
for r in B.itertuples():
    rows.append({"task": r.task, "group": r.group, "label": r.label, "phrase": r.phrase,
                 "prior_success": r.prior_success, "prior_n": r.prior_n, "source": "B",
                 "single_concept": True, "format_matched": True})

M = pd.DataFrame(rows)
# dedupe on (task, exact phrase) -- merge provenance, keep all group memberships
dup = M.groupby(["task", "phrase"])
uniq = []
for (task, phrase), g in dup:
    uniq.append({"task": task, "phrase": phrase,
                 "groups": "+".join(sorted(set(g.group))),
                 "labels": "+".join(sorted(set(g.label))),
                 "source": "+".join(sorted(set(g.source))),
                 "prior_success": g.prior_success.iloc[0],
                 "prior_n": int(g.prior_n.max()),
                 "single_concept": bool(g.single_concept.all()),
                 "format_matched": bool(g.format_matched.all())})
U = pd.DataFrame(uniq).sort_values(["task", "phrase"]).reset_index(drop=True)

GRID = {t: 24 for t in U.task.unique()}
GRID["widowx_put_eggplant_in_basket"] = 60  # enumerated 2026-09-10 (grid_sizes.json, cv2)
U["grid"] = U.task.map(GRID)
U["episodes"] = U.grid * 3

M.to_parquet(R / "results/analysis/bridge_pair_manifest_members.parquet", index=False)
U.to_parquet(R / "results/analysis/bridge_pair_manifest.parquet", index=False)
print(f"members: {len(M)} rows ({len(A)} A-groups + {B.group.nunique()} B-groups)")
print(f"unique (task, phrase) to roll: {len(U)} across {U.task.nunique()} tasks")
print(f"episodes @ full grid x3 reps: {U.episodes.sum()}")
print(U.groupby("source").size())
dupes = M.groupby(["task", "phrase"]).size()
print("cross-set dedupes:", (dupes > 1).sum())
