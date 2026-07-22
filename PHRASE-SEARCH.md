# PHRASE-SEARCH — LLM-guided rollout search over instruction phrasings

**Method.** Per task: rounds of 16 phrases proposed by Claude (informed by all
prior boards), each rolled on the SEARCH split (layouts 0–17, ×2, n=36/phrase,
SE ≈ 8pp); survivors CONFIRMED on the held-out split (layouts 18–23, ×12,
n=72/phrase). Cross-split levels shift by up to ±40pp — only within-split
contrasts are valid, and only CONFIRMED numbers are quoted as results.

**Stability flags.** **[1]** one 36-episode board (18 layouts ×2; effective
n ≈ 18 clusters; pair-Δ SE ≈ 11pp — under ~16pp is noise-range).
**[R]** re-measured on ≥2 independent boards, consistent sign. **[U]** sign or
magnitude flipped on re-measurement. **Confirmed** = held-out 6 layouts ×12.

---

## Scene by scene

### 1. Coke can → ramekin — THE rescue (+59.7 confirmed)
![](results/search/scenes/coke_can_on_ramekin.png)

**Confirmed:** "place the red coke can inside the white bowl" **95.8** vs
nominal "put coke can on ramekin" **36.1** → **+59.7. Task solved by phrasing.**

**Scene reading:** the "ramekin" is a large fluted white vessel that simply
*is* a white bowl to the eye — and the can must go IN it, while the nominal
says "on".

| pair (same frame) | values | Δ | flag |
|---|---|---|---|
| "white **bowl**" vs "white **dish**" | 63.9 vs 2.8 | **−61** | [1] most violent token swap recorded |
| "white bowl" vs "white **ramekin**" | 63.9 vs 22.2 | −42 | [1] true name loses to look-alike name |
| "**white** bowl" vs "**ceramic** bowl" | 63–72 vs 27.8 | ~−35 | [1] color ≫ material; corpus: appearance-adj+bowl ≈ 4.5k, "ceramic bowl" = 0 |
| "white bowl" vs "white **cup**" | 61.1 vs 44.4 | −17 | [R] |
| nominal's "**on** ramekin" vs in/inside-bowl forms | 0–11 vs 60–72 | ~−55 | [R] wrong relation + wrong noun |
| "**move**…into" vs "place…inside" | 22.2 vs 61.1 | −39 | [1] |
| single-clause vs pick-up-and-place | 69.4 vs 61.1 | +8 | [R] |
| ± "upright" | pooled ~+10 for dropping | — | [U→R-ish] orientation word unnecessary |

### 2. Stack cubes — modest rescue (+12.5 confirmed), brutal split shift
![](results/search/scenes/stack_cube.png)

**Confirmed:** "put the small green cube on the yellow cube" **19.4** vs
nominal "stack the green block on the yellow block" **6.9** → **+12.5**.
Held-out layouts ran ~40pp HARDER than the search split — the search-split
telegram lead (58.3) did not transfer.

**Scene reading:** both objects are literal, perfect cubes (muted teal-green
and pale yellow) — "cube" ≫ "block" is visual-fidelity naming; the honest
"teal-green" is corpus-alien.

| pair | values | Δ | flag |
|---|---|---|---|
| cube-family vs block-family | 44–58 vs 22–33 (search) ; cube forms > block nominal (confirmed) | ~+25 | **[R + confirmed]** |
| "**teal-green** cube" vs "green cube" | 22.2 vs 44–58 | ~−30 | [1] unusual compound adjective = poison |
| telegram "green cube on yellow cube" | 58.3 search → 11.1 confirmed | — | winner's-curse + split-shift case study |
| "stack the cubes" (underspecified) | 0.0 | — | [1] name both objects |

### 3. Carrot → keyboard — real modest rescue (+15.2 confirmed)
![](results/search/scenes/carrot_on_keyboard.png)

**Confirmed:** "**set** the carrot on the **black** keyboard" **31.9** vs
nominal **16.7** → **+15.2**.

**Scene reading:** slim black keyboard, low contrast, slightly raised —
balancing a rigid carrot on a narrow hard deck is mechanically hard; the
ceiling (~32) is visible in the picture.

| pair | values | Δ | flag |
|---|---|---|---|
| "black keyboard" vs bare "keyboard" | 25.0 vs ~17 | +8 | [R-ish] the one adjective that helps |
| "the **keys**" vs "the keyboard" | 2.8 vs ~17–25 | ~−17 | [1] part-name loses |
| "keyboard keys" compound | 36.1 search → **8.3 confirmed** | — | winner's-curse collapse #3 |
| "set" vs "place" (confirmed) | 31.9 vs 27.8 | +4 | confirmed boards' first "set" win |

### 4. Carrot → wheel — ceiling-limited (confirmation pending)
![](results/search/scenes/carrot_on_wheel.png)

**Best so far:** "put the carrot on the black wheel" 38.9 [1] vs nominal 22.2;
held-out confirmation in flight.

**Scene reading:** a small tire lying flat — narrow curved ring, shallow hub;
mechanically awkward for a long carrot. Low ceiling is visible.

| pair | values | Δ | flag |
|---|---|---|---|
| "black tire/wheel" vs bare | ~+8 | — | [R-ish] |
| "black tire" vs "black **rubber** tire" | 30.6 vs 16.7 | −14 | [1] color ≫ material, replication #2 |
| "the **rim**" | 5.6 | — | [1] part-name loses again |
| "in the wheel" (hub probe) | mid-band | — | [1] no in-hub advantage |

### 5. Coke can → plate — nominal unbeaten (confirmed tie)
![](results/search/scenes/coke_can_on_plate.png)

**Confirmed:** "put the coke on the plate" 84.7 vs nominal "put coke can on
plate" 83.3 — **tie**; 48 candidates found nothing better.

**Scene reading:** the plate is pale yellow-green — ambiguously "yellow" or
"green", which likely explains plate-adjective instability across tasks.

| pair | values | Δ | flag |
|---|---|---|---|
| "coke" vs "soda"/"cola"/bare "can" | 61.1 vs 47.2/44.4/30.6 | −14…−31 | [R] exact token carries 15–30pp |
| "coke" vs "**pepsi**" | 61.1 vs 58.3 | −3 | [1] wrong brand ≈ free; wrong category ≈ fatal |
| "the plate" vs "the **dish**" | 61.1 vs 30.6 | −31 | [1] |
| "the plate" vs "the **yellow** plate" | 61.1 vs 41.7 | −19 | [R] |
| "on" vs "on top of" | 61.1 vs 38.9 | −22 | [1] here only |
| "on" vs "onto" | −17 then +3 | — | **[U]** |
| "take X and put" vs "pick up X and place" | 30.6 vs 55.6 | −25 | [1] |
| "please …" | ±0 | — | [1] politeness free |
| "the coke" (no "can") | ≥ coke-can forms | +4…+8 | [R] |

### 6. Carrot → plate — nominal WINS (confirmed)
![](results/search/scenes/carrot_on_plate.png)

**Confirmed:** nominal "put carrot on plate" **48.6** beats search-best
"place the orange carrot on the green plate" 45.8. The search-split color
advantage (+17) did not survive; the r2 leader collapsed 61.1 → **30.6**
(winner's-curse case study #1, the sharpest recorded).

### 7. Spoon → towel — nominal WINS (confirmed)
![](results/search/scenes/spoon_on_towel.png)

**Confirmed:** nominal "put the spoon on the towel" **69.4** beats
"set the spoon down on the towel" 62.5.

**Scene reading:** wide, flat, forgiving landing. "blue towel" is visually
faithful yet HURT (−19 [1]) — despite "blue towel" appearing **5,815 times**
in the policy's training text: corpus frequency does not overrule a task's
dominant plain form. "towel" vs "cloth": −17 [1].

### 8. Eggplant → basket — null control (confirmed plateau)
![](results/search/scenes/put_eggplant_in_basket.png)

**Confirmed:** plateau 94.4–95.8; nominal within noise of everything.

**Scene reading:** the "yellow basket" is literally a **yellow dish rack in a
toy sink** — solving the legacy mystery cell ("aubergine…dish rack" 92.7%).
Huge open target → forgiving task → nothing for phrasing to fix.
"bin" for "basket": −39 [1]. Full telegram: −36 [1].

---

## Cross-cutting effects (replicated across ≥2 scenes)

| effect | scenes | direction & size | grade |
|---|---|---|---|
| **Exact-token identity**: near-synonyms are cliffs (coke/soda, basket/bin, towel/cloth, cube/block, bowl/dish) | 5 | −14…−61 | [R across scenes] |
| **Visual-fidelity naming**: the winning noun names what the object LOOKS like (bowl-ramekin, cube-blocks, dish-rack-basket) | 3 | decides the rescues | [R] |
| **Color ≫ material adjectives** (white/ceramic, black/rubber) + corpus template (appearance-adj+noun frequent, material-adj absent) | 2 + corpus | −14…−35 | [R] |
| **Adjectives help only where the task is hard/OOV** (black on keyboard/wheel) and hurt on good-nominal tasks (yellow plate, blue towel) | 4 | ±8…−19 | [R] |
| **Part-names lose to whole-names** (keys, rim) | 2 | −12…−17 | [R] |
| **"set" ≈ "put"**, sometimes ahead (confirmed winner on keyboard); "please" free — both wrongly banned by rules-v1 | 4+ boards | ~0…+4 | [R] |
| **Good nominal ⇒ unbeatable**: held-out crowned the nominal 4/4 times (coke-plate, carrot-plate, spoon, eggplant) | 4 | — | confirmed |
| **Winner's curse universal**: every search-split leader regressed; 4 named collapses (−10…−31) | all | — | confirmed |
| **Split-level shifts both directions** (coke-plate +20 easier held-out; stack −40 harder) | 2 | — | confirmed |

**The headroom law (final):** headroom = phrasing-conditional ceiling −
nominal. Confirmed rescues: ramekin +59.7, keyboard +15.2, stack +12.5 —
all bad-nominal tasks. All four good-nominal tasks: nominal unbeaten.

## Reading discipline
1. Per-phrase SE ≈ 8pp at n=36; singles under ~16pp are suggestive only.
2. Trust replicated bands and confirmed numbers, never single leaders.
3. Micro-rules transfer poorly across tasks; what transfers: right token,
   right relation for the geometry, plain register, trust good nominals.

*Wheel confirmation pending; its row updates on landing.*
