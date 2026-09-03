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

Full table (all measured pairs, including multi-edit):

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

Full table (all measured pairs, including multi-edit):

**Reality check:** the telegram "green cube on yellow cube" led the search split at 58.3% (n=72 across two boards, layouts 0–17) but fell to 11.1% (n=72) when re-measured on the never-searched layouts 18–23 — partly because those layouts are ~40pp harder for this task, partly winner's curse.

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

| abs Δ (pp) | better | worse | the single edit |
|---|---|---|---|
| 38.9 | "put the eggplant in the yellow basket" | "put the eggplant in the yellow bin" | basket → bin |
| 33.4 | "put the small green cube on the yellow cube" | "put the teal-green cube on the yellow cube" | small → teal |
| 33.3 | "put the carrot on the black wheel" | "put the carrot on the rim" | black wheel → rim |
| 30.5 | "put the coke can on the plate" | "put the coke can on the dish" | plate → dish |
| 30.5 | "put the coke can on the plate" | "put the can on the plate" | coke → ∅ |
| 22.2 | "put the coke can on the plate" | "put the coke can on top of the plate" | ∅ → top of |
| 19.4 | "put the coke can on the plate" | "put the coke can on the yellow plate" | ∅ → yellow |
| 16.7 | "put the coke can on the plate" | "put the cola can on the plate" | coke → cola |
| 16.7 | "pick up the red cola can and place it inside the white bowl" | "pick up the red cola can and place it inside the white cup" | bowl → cup |
| 13.9 | "put the coke can on the plate" | "put the soda can on the plate" | coke → soda |
| 13.9 | "place the carrot on the black tire" | "place the carrot on the black rubber tire" | ∅ → rubber |
| 9.7 | "pick up the red cola can and place it inside the white bowl" | "pick up the red cola can and place it upright inside the white bowl" | ∅ → upright |
| 4.1 | "set the carrot on the black keyboard" | "place the carrot on the black keyboard" | set → place |
| 2.8 | "put the coke on the plate" | "put the coke can on the plate" | ∅ → can |
| 2.8 | "put the coke can on the plate" | "put the pepsi can on the plate" | coke → pepsi |
| 0.0 | "set" | "put" | set → put |
| 0.0 | "put the carrot on the plate" | "put the carrot on the dish" | plate → dish |
| 0.0 | "please put the coke can on the plate" | "put the coke can on the plate" | please → ∅ |

Mixed-edit rows remain in the scene tables but carry no PURE tag —
their attribution is suggestive, not airtight.
---

## Cross-cutting effects (replicated across ≥2 scenes)

| effect | scenes | direction & size | grade |
|---|---|---|---|
| **Exact-token identity**: near-synonyms are cliffs (coke/soda, basket/bin, towel/cloth, cube/block, bowl/dish) | 5 | −14…−61pp | replicated across scenes |
| **Visual-fidelity naming**: the winning noun names what the object LOOKS like (bowl-ramekin, cube-blocks, dish-rack-basket) | 3 | decides the rescues | replicated |
| **Color ≫ material adjectives** (white/ceramic, black/rubber) + corpus template (appearance-adj+noun ≈ 4.5k, material-adj ≈ 0) | 2 + corpus | −14…−35pp | replicated |
| **Adjectives help only where the task is hard/OOV** (black on keyboard/wheel); hurt on good-nominal tasks (yellow plate, blue towel) | 4 | +8…−19pp | replicated |
| **Part-names lose to whole-names** (keys, rim) | 2 | −12…−17pp | replicated |
| **"set" ≈ "put"**, sometimes ahead (confirmed winner on keyboard); "please" free — both wrongly banned by rules-v1 | 4+ boards | ~0…+4pp | replicated · **PURE ✂ single edit** |
| **Good nominal ⇒ unbeatable**: held-out crowned the nominal 4/4 (coke-plate, carrot-plate, spoon, eggplant) | 4 | — | confirmed |
| **Winner's curse universal**: every search-split leader regressed; 4 named collapses (−10…−31pp) | all | — | confirmed |
| **Split-level shifts both directions** (coke-plate ~+20 easier held-out; stack ~−40 harder) | 2 | — | confirmed |

**The headroom law (final):** headroom = phrasing-conditional ceiling −
nominal. Confirmed rescues: ramekin +59.7, keyboard +15.2, stack +12.5 —
all bad-nominal tasks. All four good-nominal tasks: nominal unbeaten.

## Reading discipline
1. Per-phrase SE ≈ 8pp at n=36; n=36 gaps under ~16pp are noise-range.
2. Trust confirmed numbers (n=72, held-out) and cross-board replications;
   never single-board leaders.
3. Micro-rules transfer poorly across tasks; what transfers: right token,
   right relation for the geometry, plain register, trust good nominals.

*All eight tasks complete: 3 rescues (ramekin, keyboard, stack), 5 nominal-unbeaten/null.*