# 20 minimal-pair candidates: tiny benign edits, large success gaps

Mined 2026-09-10 from every pooled rollout leg in `results/rules_runs/r1_sim/jobs/`
(1,589 legs; 3,817 (task, phrase) cells with n>=24 episodes; sealed-12 +
val8/stock tasks; adversarial, natural, original and applier-rewrite arms).
Success = ground-truth environment flag, pooled over layouts/reps; CRN pins
layouts per (task, episode, rep), but reps and source conditions are NOT
matched across the two sides of a pair — this is a CANDIDATE list, to be
verified with matched rollouts before external quotation. z = two-proportion.
Full ranked list: `minimal_pairs_candidates.csv` (978 pairs at z>=2).

Curation: referent-CHANGING edits excluded (coke->pepsi "corrections",
wrong-color renames, next-to->over-to goal shifts); everything below keeps the
same referents and goal.

## Case / punctuation only (the most benign possible change)
| # | task | low phrase | % (n) | high phrase | % (n) | gap |
|---|------|-----------|-------|------------|-------|-----|
| 1 | pepsi_on_plate | "Set the Pepsi down on the plate." | 19.4 (432) | "set the pepsi down on the plate" | 82.3 (96) | **+62.8** (z=12.1) |
| 2 | pepsi_on_plate | "Put the Pepsi on the plate." | 18.3 (240) | "put the pepsi on the plate" | 61.5 (96) | **+43.1** (z=7.8) |

## Preposition synonym
| # | task | low phrase | % (n) | high phrase | % (n) | gap |
|---|------|-----------|-------|------------|-------|-----|
| 3 | stack_cube | "place the green block onto the yellow block" | 8.3 (24) | "place the green block on the yellow block" | 54.2 (24) | **+45.8** (z=3.4) |
| 4 | spoon_on_towel | "put the spoon on the towel" | 30.4 (69) | "put the spoon atop the towel" | 66.7 (72) | **+36.2** (z=4.3) |
| 5 | coke_can_on_wheel | "Put the red can on the black wheel." | 4.2 (24) | "Put the red can into the black wheel." | 31.2 (96) | **+27.1** (z=2.7) |
| 6 | carrot_on_ramekin | "Place the carrot in the ramekin." | 19.3 (192) | "Place the carrot inside the ramekin." | 40.6 (96) | **+21.4** (z=3.9) |
| 7 | pepsi_on_plate | "Move the blue can to the plate." | 16.7 (48) | "Move the blue can onto the plate." | 37.5 (168) | **+20.8** (z=2.7) |

## Verb synonym
| # | task | low phrase | % (n) | high phrase | % (n) | gap |
|---|------|-----------|-------|------------|-------|-----|
| 8 | nut_on_wheel | "Position the nut right on the wheel." | 4.2 (120) | "Put the nut right on the wheel." | 45.8 (48) | **+41.7** (z=6.6) |
| 9 | cube_on_plate | "put the green cube into the dish" | 37.5 (24) | "place the green cube into the dish" | 75.0 (48) | **+37.5** (z=3.1) |
| 10 | eggplant_on_sponge | "Place the purple eggplant onto the yellow and green sponge." | 60.4 (48) | "Deposit the purple eggplant onto the yellow and green sponge." | 91.7 (24) | **+31.2** (z=2.8) |
| 11 | eggplant_on_sponge | "Move the purple eggplant onto the sponge." | 59.5 (168) | "Set the purple eggplant onto the sponge." | 89.6 (48) | **+30.1** (z=3.9) |
| 12 | eggplant_on_keyboard | "Move the purple eggplant on the black keyboard." | 16.7 (24) | "Rest the purple eggplant on the black keyboard." | 45.8 (24) | **+29.2** (z=2.2) |

## Noun synonym (same referent)
| # | task | low phrase | % (n) | high phrase | % (n) | gap |
|---|------|-----------|-------|------------|-------|-----|
| 13 | eggplant_on_sponge | "set the purple thing on the yellow and green sponge" | 22.9 (96) | "set the purple eggplant on the yellow and green sponge" | 72.9 (96) | **+50.0** (z=6.9) |
| 14 | pepsi_on_plate | "set the soda on the plate" | 29.2 (24) | "set the pepsi on the plate" | 75.0 (48) | **+45.8** (z=3.7) |
| 15 | coke_can_on_keyboard | "Put the red soda can on top of the black keyboard." | 2.8 (72) | "Put the red soda cylinder on top of the black keyboard." | 45.8 (48) | **+43.1** (z=5.8) |
| 16 | carrot_on_ramekin | "Pick up the orange vegetable and place it in the white bowl." | 12.5 (24) | "Pick up the orange carrot and place it in the white bowl." | 54.2 (48) | **+41.7** (z=3.4) |
| 17 | coke_can_on_plate | "put the coke can on the dish" | 0.0 (60) | "put the coke can on the plate" | 38.1 (105) | **+38.1** (z=5.5) |

## Slight restructure
| # | task | low phrase | % (n) | high phrase | % (n) | gap |
|---|------|-----------|-------|------------|-------|-----|
| 18 | eggplant_on_sponge | "purple Eggplant set on the sponge." | 45.8 (24) | "Set the purple eggplant on the sponge." | 83.3 (48) | **+37.5** (z=3.3) |
| 19 | eggplant_on_keyboard | "put the eggplant on the keyboard" | 12.0 (192) | "put eggplant on keyboard" | 37.5 (672) | **+25.5** (z=6.7) |
| 20 | eggplant_on_sponge | "put the purple eggplant on the yellow and green sponge" | 62.1 (240) | "please put the purple eggplant on the green and yellow sponge" | 83.3 (24) | **+21.3** (z=2.1) |

## Notes
- #1-2: capitalization/punctuation ONLY (identical words). 79 such pairs found
  at z>=2 — brand-case "Pepsi"->"pepsi" alone moves +40pp (n=96/96). The
  tokenizer, not the semantics, owns much of the adversarial surface.
- Directions are not consistent across tasks: "please" helps eggplant (#20)
  but costs the pepsi task ~46pp (style-confounded variant); "atop" BEATS "on"
  (#4) while "onto" LOSES to "on" (#3); article-dropping helps in-vocab
  canonicals (#19, the corpus-exact form). Simple edits, both signs.
- #19-20 also carry a color-order/article confound alongside the stated edit.
- Verification protocol when needed: matched CRN rollouts, both phrases of a
  pair at 24 layouts x 2 reps in one leg (~$1/pair on one 4090).
