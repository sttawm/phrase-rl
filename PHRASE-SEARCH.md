# PHRASE-SEARCH — LLM-guided rollout search over instruction phrasings

**Method.** Per task: rounds of 16 phrases proposed by Claude (informed by
all prior boards), rolled on layouts 0–17 (×2 = 36 episodes/board); leading
phrases re-measured on the never-searched layouts 18–23 (×12 = 72 episodes).
**Every number below pools all measurements of that phrase across both layout
sets, with the combined n labeled.** The two layout sets differ in difficulty
(by up to ±40pp on some tasks), so a pair whose two sides have very different
n-mixes can carry residual split bias — the per-scene "reality check" notes
flag where that mattered; raw per-split values live in
results/search/*_results.json. Per-phrase SE at n=36 ≈ 8pp, n=108 ≈ 4.5pp.

**Purification rounds (`*_pure1`).** After the main search, five scenes got a
minimal-pair round: every phrase shares ONE frame and each row differs from
the board's control by exactly one edit, all rolled on the same layouts
(0–17 ×2, n=36 each). These are the cleanest attribution numbers in this
file — same split, same frame, single edit — and where they contradict the
pooled mixed-edit tables, **the purified number wins**. Several headline
"cliffs" below were revised this way (dish, ceramic, cloth, blue).

---

## Scene by scene

### 1. Coke can → ramekin — THE rescue (+59.7 confirmed)
![](results/search/scenes/coke_can_on_ramekin.png)

**Verdict (all data pooled):** "place the red coke can inside the white bowl" **87.9%** (n=108) vs nominal "put coke can on ramekin" **24.1%** (n=108) → **+63.9pp**. Solved by phrasing.
**Scene reading:** the "ramekin" is a large fluted white vessel that simply
*is* a white bowl to the eye — and the can must go IN it, while the nominal
says "on".

**Highlights — single-concept edits (the only difference bolded):**

| Δ (pp) | better | success | worse | success |
|---|---|---|---|---|
| **+20.4** | "pick up the red cola can and place it inside the white **bowl**" | 64.8% (n=108) | "pick up the red cola can and place it inside the white **cup**" | 44.4% (n=36) |
| **+7.8** | "pick up the red cola can and place it inside the white bowl" | 64.8% (n=108) | "pick up the red cola can and place it **upright** inside the white bowl" | 57.0% (n=72) |

**Purified minimal pairs** — one frame, control = "place the red cola can inside the white **bowl**" **61.1%**, each row one edit (n=36/row):

| Δ vs control (pp) | phrase | success | the single edit |
|---|---|---|---|
| **+16.7** | "place the red cola can inside the white **basin**" | 77.8% | bowl → basin |
| **+11.1** | "place the red cola can inside the white **container**" | 72.2% | bowl → container |
| **+8.3** | "place the red cola can inside the **ceramic** bowl" | 69.4% | white → ceramic |
| **+8.3** | "place the red cola can **on** the white bowl" | 69.4% | inside → on |
| +2.8 | "place the red cola can inside the white **cup**" | 63.9% | bowl → cup |
| +2.8 | "place the red cola can **into** the white bowl" | 63.9% | inside → into |
| +2.8 | "**put** the red cola can inside the white bowl" | 63.9% | place → put |
| 0.0 | "place the red cola can inside the bowl" / "**set** …" | 61.1% | drop white / place → set |
| −5.6 | "place the red cola can **in** the white bowl" | 55.6% | inside → in |
| −8.3 | "**move** the red cola can inside the white bowl" | 52.8% | place → move |
| −8.3 | "place the red cola can inside the white **dish**" | 52.8% | bowl → dish |
| −11.1 | "place the red cola can inside the white **pot**" | 50.0% | bowl → pot |
| **−13.9** | "place the red **soda** can inside the white bowl" | 47.2% | cola → soda |
| **−41.7** | "place the red cola can inside the white **ramekin**" | 19.4% | bowl → ramekin (TRUE NAME) |

**What purification overturned:** the giant "dish" (−54) and "ceramic" (−35)
cliffs in the mixed table below were stacked-edit artifacts (bare-can and
soda+upright confounds); at a single edit, dish is −8.3 and ceramic is a
harmless +8.3. Category words (container, basin) are FINE — basin *beats*
bowl. Relation words (in/into/inside/on) are all within noise. What
survives clean: the object's rare TRUE NAME is catastrophic (−41.7) and the
soda token costs −13.9 (replicating coke-plate's soda −8.3/−13.9).

Full table (all measured pairs, including multi-edit — superseded where the purified rows above disagree):

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+37.0** | "put the coke can in the white bowl" | 61.1% (n=72) | "put coke can on ramekin" (nominal) | 24.1% (n=108) | in-bowl → on-ramekin |
| **+54.2** | "Pick up the red cola can and place it upright inside the white bowl." | 57.0% (n=72) | "pick up the can and place it upright inside the white dish" | 2.8% (n=36) | bowl → dish |
| **+35.2** | "place the red cola can inside the white bowl" | 57.4% (n=108) | "move the red can into the white bowl" | 22.2% (n=36) | place → move |
| **+37.0** | "pick up the red cola can and place it inside the white bowl" | 64.8% (n=108) | "pick up the soda can and place it upright inside the ceramic bowl." | 27.8% (n=36) | white → ceramic (+ soda) |
| **+34.8** | "Pick up the red cola can and place it upright inside the white bowl." | 57.0% (n=72) | "place the coke can inside the white ramekin" | 22.2% (n=36) | bowl → ramekin (true name) |
| **+20.4** | "pick up the red cola can and place it inside the white bowl" | 64.8% (n=108) | "pick up the red cola can and place it inside the white cup" | 44.4% (n=36) | bowl → cup · **PURE ✂ single edit** |
| **+7.8** | "pick up the red cola can and place it inside the white bowl" | 64.8% (n=108) | "pick up the red cola can and place it upright inside the white bowl" | 57.0% (n=72) | ± "upright" · **PURE ✂ single edit** |
| **−7.4** | "place the red cola can inside the white bowl" (single clause) | 57.4% (n=108) | "pick up the red cola can and place it inside the white bowl" | 64.8% (n=108) | drop first clause |

### 2. Stack cubes — modest rescue (+12.5 confirmed), brutal split shift
![](results/search/scenes/stack_cube.png)

**Verdict (all data pooled):** "put the small green cube on the yellow cube" **36.8%** (n=144) vs nominal "stack the green block on the yellow block" **12.0%** (n=108) → **+24.8pp**. Modest rescue.
telegram lead did not transfer.

**Scene reading:** both objects are literal, perfect cubes (muted teal-green
and pale yellow) — "cube" ≫ "block" is visual-fidelity naming; the honest
"teal-green" is corpus-alien.

**Highlights — single-concept edits (the only difference bolded):**

| Δ (pp) | better | success | worse | success |
|---|---|---|---|---|
| **+14.6** | "put the **small** **green** cube on the yellow cube" | 36.8% (n=144) | "put the **teal-green** cube on the yellow cube" | 22.2% (n=36) |

**Purified minimal pairs** — one frame, control = "put the small green cube on the yellow cube" **52.8%**, each row one edit (n=36/row):

| Δ vs control (pp) | phrase | success | the single edit |
|---|---|---|---|
| **−13.9** | "put the green cube on the yellow cube" | 38.9% | drop "small" |
| **−19.4** | "**stack** the small green cube on the yellow cube" | 33.3% | put → stack (the verb alone!) |
| **−25.0** | "put the small green **block** on the yellow **block**" | 27.8% | cube → block |
| **−33.4** | "put the small **teal-green** cube on the yellow cube" | 19.4% | green → teal-green |

The nominal's own verb "stack" costs −19.4 fully specified. "Cube ≫ block"
is confirmed clean (−25.0; corpus: cube ×407, block ×159). The visually
honest "teal-green" is the worst single edit on the board (−33.4): the
corpus's plain color word beats visual precision — visual-fidelity naming
works only *within* the training vocabulary. And the size cue "small"
carries real signal (−13.9 when dropped).

Full table (all measured pairs, including multi-edit):

**Reality check:** the telegram "green cube on yellow cube" led the search split at 58.3% (n=72 across two boards, layouts 0–17) but fell to 11.1% (n=72) when re-measured on the never-searched layouts 18–23 — partly because those layouts are ~40pp harder for this task, partly winner's curse. Same story for the purified control itself: 52.8% on the search split above vs 19.4% (n=72) on the confirm split.

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+27.1** | "put the green cube on the yellow cube" | 27.1% (n=144) | "stack the cubes" | 0.0% (n=36) | both objects named → underspecified |
| **+14.6** | "put the small green cube on the yellow cube" | 36.8% (n=144) | "put the teal-green cube on the yellow cube" | 22.2% (n=36) | plain color → compound color · **PURE ✂ single edit** |
| **+15.1** | "put the green cube on the yellow cube" | 27.1% (n=144) | "stack the green block on the yellow block" (nominal) | 12.0% (n=108) | cube → block |

### 3. Carrot → keyboard — real modest rescue (+15.2 confirmed)
![](results/search/scenes/carrot_on_keyboard.png)

**Verdict (all data pooled):** "set the carrot on the black keyboard" **31.5%** (n=108) vs nominal "put carrot on keyboard" **16.7%** (n=108) → **+14.8pp**. Modest rescue.
**Scene reading:** slim black keyboard, low contrast, slightly raised —
balancing a rigid carrot on a narrow hard deck is mechanically hard; the
ceiling (~32) is visible in the picture.

**Highlights — single-concept edits (the only difference bolded):**

| Δ (pp) | better | success | worse | success |
|---|---|---|---|---|
| **+5.1** | "**set** the carrot on the black keyboard" | 31.5% (n=108) | "**place** the carrot on the black keyboard" | 26.4% (n=144) |

Full table (all measured pairs, including multi-edit):

**Reality check:** "place the carrot on the keyboard keys" scored 36.1% (n=36) on the search layouts but only 8.3% (n=72) on the held-out layouts — a winner's-curse collapse; it was never a real winner.

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+23.6** | "place the carrot on the black keyboard" | 26.4% (n=144) | "put the carrot on the keys" | 2.8% (n=36) | whole-name → part-name |
| **+9.7** | "place the carrot on the black keyboard" | 26.4% (n=144) | "put carrot on keyboard" (nominal) | 16.7% (n=108) | +black, place-frame |
| **+5.1** | "set the carrot on the black keyboard" | 31.5% (n=108) | "place the carrot on the black keyboard" | 26.4% (n=144) | set → place · **PURE ✂ single edit** |

**Purified minimal pairs** — one frame, control = "set the carrot on the black keyboard" **30.6%**, each row one edit (n=36/row):

| Δ vs control (pp) | phrase | success | the single edit |
|---|---|---|---|
| −2.8 | "**put** the carrot on the black keyboard" | 27.8% | set → put (noise) |
| **−11.2** | "set the carrot on the keyboard" | 19.4% | drop "black" |

The color adjective is doing real work on this OOV object (−11.2 dropped);
set vs put is noise here.

### 4. Carrot → wheel — ceiling-limited, NO rescue (confirmed)
![](results/search/scenes/carrot_on_wheel.png)

**Verdict (all data pooled):** "put the carrot on the black wheel" **32.7%** (n=144) vs nominal "put carrot on wheel" **22.9%** (n=144) → **+9.8pp**. No rescue — within noise; ceiling-limited.

**Scene reading:** a small tire lying flat — narrow curved ring, shallow hub;
mechanically awkward for a long carrot. Low ceiling visible.

**Highlights — single-concept edits (the only difference bolded):**

| Δ (pp) | better | success | worse | success |
|---|---|---|---|---|
| **+29.2** | "put the carrot on the **black** **wheel**" | 34.8% (n=72) | "put the carrot on the **rim**" | 5.6% (n=36) |
| **+6.9** | "place the carrot on the black tire" | 23.6% (n=72) | "place the carrot on the black **rubber** tire" | 16.7% (n=36) |

Full table (all measured pairs, including multi-edit):

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+29.2** | "put the carrot on the black wheel" | 34.8% (n=72) | "put the carrot on the rim" | 5.6% (n=36) | whole-name → part-name · **PURE ✂ single edit** |
| **+15.3** | "place the carrot on the black tire" | 23.6% (n=72) | "put the carrot on the tire" | 8.3% (n=36) | +black, place-frame |
| **+6.9** | "place the carrot on the black tire" | 23.6% (n=72) | "place the carrot on the black rubber tire" | 16.7% (n=36) | +material adjective · **PURE ✂ single edit** |

### 5. Coke can → plate — nominal unbeaten (confirmed tie)
![](results/search/scenes/coke_can_on_plate.png)

**Verdict (all data pooled):** "put the coke on the plate" **75.0%** (n=144) vs nominal "put coke can on plate" **70.0%** (n=180) → **+5.0pp**. Nominal unbeaten (tie); 48 candidates tried.

**Scene reading:** the plate is pale yellow-green — ambiguously "yellow" or
"green", which likely explains plate-adjective instability across tasks.

**Highlights — single-concept edits (the only difference bolded):**

| Δ (pp) | better | success | worse | success |
|---|---|---|---|---|
| **+24.9** | "put the **coke** can on the plate" | 55.5% (n=108) | "put the can on the plate" | 30.6% (n=36) |
| **+24.9** | "put the coke can on the **plate**" | 55.5% (n=108) | "put the coke can on the **dish**" | 30.6% (n=36) |
| **+16.6** | "put the coke can on the plate" | 55.5% (n=108) | "put the coke can on **top** **of** the plate" | 38.9% (n=36) |
| **+13.8** | "put the coke can on the plate" | 55.5% (n=108) | "put the coke can on the **yellow** plate" | 41.7% (n=36) |
| **+11.1** | "put the **coke** can on the plate" | 55.5% (n=108) | "put the **cola** can on the plate" | 44.4% (n=36) |
| **+8.3** | "put the **coke** can on the plate" | 55.5% (n=108) | "put the **soda** can on the plate" | 47.2% (n=36) |
| **−2.8** | "put the **coke** can on the plate" | 55.5% (n=108) | "put the **pepsi** can on the plate" | 58.3% (n=36) |
| **+19.5** | "put the coke on the plate" | 75.0% (n=144) | "put the coke **can** on the plate" | 55.5% (n=108) |
| **+5.6** | "**please** put the coke can on the plate" | 61.1% (n=36) | "put the coke can on the plate" | 55.5% (n=108) |

Full table (all measured pairs, including multi-edit):

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+24.9** | "put the coke can on the plate" | 55.5% (n=108) | "put the can on the plate" | 30.6% (n=36) | drop "coke" entirely · **PURE ✂ single edit** |
| **+24.9** | "put the coke can on the plate" | 55.5% (n=108) | "put the coke can on the dish" | 30.6% (n=36) | plate → dish · **PURE ✂ single edit** |
| **+25.0** | "pick up the coke can and place it on the plate" | 55.6% (n=36) | "take the coke can and put it on the plate" | 30.6% (n=36) | pick-up → take |
| **+16.6** | "put the coke can on the plate" | 55.5% (n=108) | "put the coke can on top of the plate" | 38.9% (n=36) | on → on top of · **PURE ✂ single edit** |
| **+13.8** | "put the coke can on the plate" | 55.5% (n=108) | "put the coke can on the yellow plate" | 41.7% (n=36) | +yellow · **PURE ✂ single edit** |
| **+11.1** | "put the coke can on the plate" | 55.5% (n=108) | "put the cola can on the plate" | 44.4% (n=36) | coke → cola · **PURE ✂ single edit** |
| **+8.3** | "put the coke can on the plate" | 55.5% (n=108) | "put the soda can on the plate" | 47.2% (n=36) | coke → soda · **PURE ✂ single edit** |
| **−5.1** | "set the coke can on the plate" | 64.8% (n=108) | "put coke can on plate" (nominal) | 70.0% (n=180) | set ≈ put (within noise) |
| **−2.8** | "put the coke can on the plate" | 55.5% (n=108) | "put the pepsi can on the plate" | 58.3% (n=36) | coke → pepsi (wrong brand ≈ free) · **PURE ✂ single edit** |
| **+19.5** | "put the coke on the plate" | 75.0% (n=144) | "put the coke can on the plate" | 55.5% (n=108) | dropping "can" costs nothing · **PURE ✂ single edit** |
| **+5.6** | "please put the coke can on the plate" | 61.1% (n=36) | "put the coke can on the plate" | 55.5% (n=108) | "please" is free · **PURE ✂ single edit** |

**Purified minimal pairs (clause structure)** — one frame, control = "put the coke can on the plate" **63.9%**, each row one edit from its neighbor (n=36/row):

| Δ (pp) | better phrase | success | worse phrase | success | the single edit |
|---|---|---|---|---|---|
| −5.6 | "put the coke can on the plate" | 63.9% | "pick up the coke can and **place** it on the plate" | 58.3% | one clause → two clauses (noise) |
| **+19.4** | "pick up the coke can and **place** it on the plate" | 58.3% | "pick up the coke can and **put** it on the plate" | 38.9% | place → put as the 2nd verb |
| **+22.2** | "**pick up** the coke can and place it on the plate" | 58.3% | "**take** the coke can and place it on the plate" | 36.1% | pick up → take |

Adding a pick-up clause is free — but the WORDS inside the clause matter
enormously: "place" ≫ "put" as the placement verb (+19.4) and "pick up" ≫
"take" (+22.2). The corpus's dominant two-clause template is
"pick up X and place it on Y" — deviate from its exact verbs and it costs
~20pp each.

### 6. Carrot → plate — nominal WINS (confirmed)
![](results/search/scenes/carrot_on_plate.png)

**Verdict (all data pooled):** "place the orange carrot on the green plate" **51.4%** (n=144) vs nominal "put carrot on plate" **45.1%** (n=144) → **+6.2pp**. Nominal wins.

**Highlights — single-concept edits (the only difference bolded):**

| Δ (pp) | better | success | worse | success |
|---|---|---|---|---|
| **+0.0** | "put the carrot on the **plate**" | 38.9% (n=36) | "put the carrot on the **dish**" | 38.9% (n=36) |

Full table (all measured pairs, including multi-edit):

**Reality check:** "put the orange carrot on top of the plate" hit 61.1% (n=36) on the search layouts but 30.6% (n=72) held-out — the sharpest winner's-curse collapse recorded.

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+12.5** | "place the orange carrot on the green plate" | 51.4% (n=144) | "put the carrot on the plate" | 38.9% (n=36) | +colors (search split only) |
| **−6.2** | "put carrot on plate" (nominal) | 45.1% (n=144) | "place the orange carrot on the green plate" | 51.4% (n=144) | color gain did NOT survive |
| **+0.0** | "put the carrot on the plate" | 38.9% (n=36) | "put the carrot on the dish" | 38.9% (n=36) | dish HARMLESS here (cf. scene 5) · **PURE ✂ single edit** |

### 7. Spoon → towel — nominal WINS (confirmed)
![](results/search/scenes/spoon_on_towel.png)

**Verdict (all data pooled):** "set the spoon down on the towel" **65.7%** (n=108) vs nominal "put the spoon on the towel" **57.6%** (n=144) → **+8.1pp**. Nominal wins.

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+19.4** | "put the spoon on top of the towel" | 58.3% (n=36) | "put the spoon on the blue towel" | 38.9% (n=36) | +blue — despite "blue towel" ×5,815 in training text |
| **+13.9** | "put the spoon on top of the towel" | 58.3% (n=36) | "put the spoon on the cloth" | 44.4% (n=36) | towel → cloth |

**Purified minimal pairs** — one frame, control = "put the spoon on the towel" **47.2%**, each row one edit (n=36/row):

| Δ vs control (pp) | phrase | success | the single edit |
|---|---|---|---|
| **+13.9** | "**set** the spoon on the towel" | 61.1% | put → set |
| +2.8 | "put the spoon **onto** the towel" | 50.0% | on → onto (harmless) |
| −2.8 | "put the spoon on the **cloth**" | 44.4% | towel → cloth (noise!) |
| −5.6 | "put the spoon on the **blue** towel" | 41.7% | +blue (mild) |

**What purification overturned:** the "cloth cliff" (−13.9 above) and the
big blue penalty (−19.4 above) were carried by the on-top-of comparator —
at a single edit, cloth is −2.8 (noise) and blue only −5.6. The real signal
on this board is "set" +13.9 over "put".

**Reality check:** the search leader "set the spoon down on the towel" (72.2%, n=36) regressed to 62.5% (n=72) on the held-out layouts — still good, but below the nominal's 69.4%.

### 8. Eggplant → basket — null control (confirmed plateau)
![](results/search/scenes/put_eggplant_in_basket.png)

**Verdict (all data pooled):** "place the purple eggplant into the yellow basket" **96.3%** (n=108) vs nominal "put eggplant into yellow basket" **93.5%** (n=108) → **+2.8pp**. Null control — saturated plateau.

**Scene reading:** the "yellow basket" is literally a **yellow dish rack in a
toy sink** — solving the legacy mystery cell ("aubergine…dish rack" 92.7%).
Huge open target → nothing for phrasing to fix.

**Highlights — single-concept edits (the only difference bolded):**

| Δ (pp) | better | success | worse | success |
|---|---|---|---|---|
| **+37.0** | "put the eggplant in the yellow **basket**" | 95.3% (n=108) | "put the eggplant in the yellow **bin**" | 58.3% (n=36) |

Full table (all measured pairs, including multi-edit):

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+37.0** | "put the eggplant in the yellow basket" | 95.3% (n=108) | "put the eggplant in the yellow bin" | 58.3% (n=36) | basket → bin · **PURE ✂ single edit** |
| **+34.2** | "put the eggplant in the yellow basket" | 95.3% (n=108) | "eggplant in yellow basket" | 61.1% (n=36) | full sentence → telegram |


## The cleanest evidence — single-edit pairs (one concept changed, everything else identical)

Rows marked `pure1` come from the single-frame purification boards
(same frame, same split, one edit; n=36 per side, SE of the difference ≈ 11pp).

| abs Δ (pp) | better | worse | the single edit |
|---|---|---|---|
| 41.7 | "place the red cola can inside the white bowl" | "…inside the white ramekin" | bowl → ramekin (TRUE NAME) · pure1 |
| 38.9 | "put the eggplant in the yellow basket" | "put the eggplant in the yellow bin" | basket → bin |
| 33.4 | "put the small green cube on the yellow cube" | "put the small teal-green cube on the yellow cube" | green → teal-green · pure1 |
| 33.3 | "put the carrot on the black wheel" | "put the carrot on the rim" | black wheel → rim |
| 30.5 | "put the coke can on the plate" | "put the coke can on the dish" | plate → dish |
| 30.5 | "put the coke can on the plate" | "put the can on the plate" | coke → ∅ |
| 25.0 | "put the small green cube on the yellow cube" | "put the small green block on the yellow block" | cube → block · pure1 |
| 22.2 | "put the coke can on the plate" | "put the coke can on top of the plate" | ∅ → top of |
| 22.2 | "pick up the coke can and place it on the plate" | "take the coke can and place it on the plate" | pick up → take · pure1 |
| 19.4 | "put the coke can on the plate" | "put the coke can on the yellow plate" | ∅ → yellow |
| 19.4 | "pick up the coke can and place it on the plate" | "pick up the coke can and put it on the plate" | place → put (2nd verb) · pure1 |
| 19.4 | "put the small green cube on the yellow cube" | "stack the small green cube on the yellow cube" | put → stack · pure1 |
| 16.7 | "put the coke can on the plate" | "put the cola can on the plate" | coke → cola |
| 16.7 | "pick up the red cola can and place it inside the white bowl" | "…inside the white cup" | bowl → cup |
| 16.7 | "place the red cola can inside the white basin" | "…inside the white bowl" | bowl → basin (basin WINS) · pure1 |
| 13.9 | "put the coke can on the plate" | "put the soda can on the plate" | coke → soda |
| 13.9 | "place the red cola can inside the white bowl" | "place the red soda can inside the white bowl" | cola → soda · pure1 |
| 13.9 | "put the small green cube on the yellow cube" | "put the green cube on the yellow cube" | drop "small" · pure1 |
| 13.9 | "set the spoon on the towel" | "put the spoon on the towel" | put → set (set WINS) · pure1 |
| 13.9 | "place the carrot on the black tire" | "place the carrot on the black rubber tire" | ∅ → rubber |
| 11.2 | "set the carrot on the black keyboard" | "set the carrot on the keyboard" | drop "black" · pure1 |
| 11.1 | "place the red cola can inside the white container" | "…inside the white bowl" | bowl → container (container wins) · pure1 |
| 9.7 | "pick up the red cola can and place it inside the white bowl" | "…place it upright inside the white bowl" | ∅ → upright |
| 4.1 | "set the carrot on the black keyboard" | "place the carrot on the black keyboard" | set → place |
| 2.8 | "put the coke on the plate" | "put the coke can on the plate" | ∅ → can |
| 2.8 | "put the coke can on the plate" | "put the pepsi can on the plate" | coke → pepsi |
| 2.8 | "put the spoon onto the towel" | "put the spoon on the towel" | on → onto (harmless) · pure1 |
| 2.8 | "put the spoon on the towel" | "put the spoon on the cloth" | towel → cloth (REVISED: noise, was "cliff") · pure1 |
| 0.0 | "put the carrot on the plate" | "put the carrot on the dish" | plate → dish |
| 0.0 | "please put the coke can on the plate" | "put the coke can on the plate" | please → ∅ |

Mixed-edit rows remain in the scene tables but carry no PURE tag —
their attribution is suggestive, not airtight.
---

## Cross-cutting effects (replicated across ≥2 scenes)

| effect | scenes | direction & size | grade |
|---|---|---|---|
| **Exact-token identity**: near-synonym swaps are cliffs (coke/soda ×2, basket/bin, cube/block, cola/soda) | 4 | −14…−39pp | replicated, survives purification |
| **TRUE-NAME toxicity**: naming a rare object by its real name when a corpus-common look-alike fits (ramekin vs bowl) | 1 (largest clean effect) | −41.7pp | pure1 |
| **Category words are FINE, hyper-specific compounds are toxic**: container/basin ≥ bowl (+11/+17) but teal-green −33; specificity must stay inside corpus vocabulary | 2 | +17…−33pp | pure1 |
| **Visual-fidelity naming**: the winning noun names what the object LOOKS like (bowl-ramekin, cube-blocks, dish-rack-basket) — *in the corpus's own words* (teal-green fails) | 3 | decides the rescues | replicated, refined by pure1 |
| **Clause-template verb identity**: extra pick-up clause free, but inside it "place" ≫ "put" (+19.4) and "pick up" ≫ "take" (+22.2) — the corpus template's exact verbs | 1 | ±20pp | pure1 |
| **Adjectives help on hard/OOV objects** (black on keyboard −11.2 dropped · pure1; black on wheel); mild-to-harmful on good-nominal tasks (yellow plate −19; blue towel −5.6 pure1) | 4 | +11…−19pp | replicated |
| **Part-names lose to whole-names** (keys, rim) | 2 | −12…−29pp | replicated |
| **"set" ≥ "put"**: +13.9 (spoon pure1), +2.8 (keyboard pure1), 0.0 (ramekin pure1), −5.1 (coke-plate pooled) — never a big loss; "please" free. Both wrongly banned by rules-v1 | 4 | +14…−5pp | replicated · pure1 |
| **Relation words mostly free**: in/into/inside/on all within noise (ramekin pure1), onto +2.8 (spoon pure1); exception "on top of" −22.2 (coke-plate) — xcert certifying now | 3 | ±3 (one −22) | pure1, certification running |
| **Verb "stack" costs −19.4 even fully specified** (put → stack, pure1); "move" −8.3 (ramekin pure1) | 2 | −8…−19pp | pure1 |
| **Dish/ceramic/cloth "cliffs" were confound artifacts**: at a single edit dish −8.3 (ramekin; but −30.5 on coke-plate — task-dependent), ceramic +8.3, cloth −2.8 | 3 | revised | pure1 overturned |
| **Good nominal ⇒ unbeatable**: held-out crowned the nominal 4/4 (coke-plate, carrot-plate, spoon, eggplant) | 4 | — | confirmed |
| **Winner's curse universal**: every search-split leader regressed; 4 named collapses (−10…−31pp) | all | — | confirmed |
| **Split-level shifts both directions** (coke-plate ~+20 easier held-out; stack ~−40 harder — stack pure control 52.8 vs 19.4 confirm) | 2 | — | confirmed |

**The headroom law (final):** headroom = phrasing-conditional ceiling −
nominal. Confirmed rescues: ramekin +59.7, keyboard +15.2, stack +12.5 —
all bad-nominal tasks. All four good-nominal tasks: nominal unbeaten.

## Reading discipline
1. Per-phrase SE ≈ 8pp at n=36; n=36 gaps under ~16pp are noise-range.
2. Trust confirmed numbers (n=72, held-out) and cross-board replications;
   never single-board leaders.
3. Micro-rules transfer poorly across tasks; what transfers: right token,
   right relation for the geometry, plain register, trust good nominals.

*All eight tasks complete: 3 rescues (ramekin, keyboard, stack), 5
nominal-unbeaten/null. Purification round complete on 5 scenes (ramekin,
stack, keyboard, coke-plate clauses, spoon). Cross-task certification
(xcert: 6 edits × 8 tasks × n=144; xcert2: clause pairs × 8 tasks) is
rolling now — its numbers supersede everything here where they overlap.*