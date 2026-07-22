# PHRASE-SEARCH — LLM-guided rollout search over instruction phrasings

**Method.** Per task: rounds of 16 phrases proposed by Claude (informed by all
prior boards), each rolled on the SEARCH split (layouts 0–17 × 2 repeats =
**n=36 episodes/phrase**, SE ≈ 8pp); survivors CONFIRMED on the held-out split
(layouts 18–23 × 12 repeats = **n=72 episodes/phrase**). Cross-split levels
shift by up to ±40pp — only within-split contrasts are valid; only confirmed
numbers are quoted as results. Every success cell below is labeled with its n.
Pairs with n=36 sides and a gap under ~16pp are within noise. Where the two
phrases differ in more than the highlighted feature, both are quoted verbatim
so the reader can judge.

---

## Scene by scene

### 1. Coke can → ramekin — THE rescue (+59.7 confirmed)
![](results/search/scenes/coke_can_on_ramekin.png)

**Confirmed (n=72):** "place the red coke can inside the white bowl" **95.8%**
vs nominal "put coke can on ramekin" **36.1%** → **+59.7pp. Solved by phrasing.**

**Scene reading:** the "ramekin" is a large fluted white vessel that simply
*is* a white bowl to the eye — and the can must go IN it, while the nominal
says "on".

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+63.9** | "put the coke can in the white bowl" | 63.9% (n=36) | "put coke can on ramekin" (nominal) | 0.0% (n=36) | in-bowl → on-ramekin |
| **+52.8** | "Pick up the red cola can and place it upright inside the white bowl." | 55.6% (n=36) | "pick up the can and place it upright inside the white dish" | 2.8% (n=36) | bowl → dish |
| **+47.2** | "place the red cola can inside the white bowl" | 69.4% (n=36) | "move the red can into the white bowl" | 22.2% (n=36) | place → move |
| **+44.4** | "pick up the red cola can and place it inside the white bowl" | 72.2% (n=36) | "pick up the soda can and place it upright inside the ceramic bowl." | 27.8% (n=36) | white → ceramic (+ soda) |
| **+33.4** | "Pick up the red cola can and place it upright inside the white bowl." | 55.6% (n=36) | "place the coke can inside the white ramekin" | 22.2% (n=36) | bowl → ramekin (true name) |
| **+16.7** | "pick up the red cola can and place it inside the white bowl" | 61.1% (n=36) | "pick up the red cola can and place it inside the white cup" | 44.4% (n=36) | bowl → cup · **PURE ✂ single edit** |
| **+9.7** | "pick up the red cola can and place it inside the white bowl" | 66.7% (n=72) | "pick up the red cola can and place it upright inside the white bowl" | 57.0% (n=72) | ± "upright" · **PURE ✂ single edit** |
| **+8.3** | "place the red cola can inside the white bowl" (single clause) | 69.4% (n=36) | "pick up the red cola can and place it inside the white bowl" | 61.1% (n=36) | drop first clause |

### 2. Stack cubes — modest rescue (+12.5 confirmed), brutal split shift
![](results/search/scenes/stack_cube.png)

**Confirmed (n=72):** "put the small green cube on the yellow cube" **19.4%**
vs nominal "stack the green block on the yellow block" **6.9%** → **+12.5pp**.
Held-out layouts ran ~40pp harder than the search split; the search-split
telegram lead did not transfer.

**Scene reading:** both objects are literal, perfect cubes (muted teal-green
and pale yellow) — "cube" ≫ "block" is visual-fidelity naming; the honest
"teal-green" is corpus-alien.

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+47.2** | "green cube on yellow cube" (telegram) | 58.3% (n=72) | same phrase, held-out | 11.1% (n=72) | split shift + winner's curse |
| **+44.4** | "put the green cube on the yellow cube" | 44.4% (n=36) | "stack the cubes" | 0.0% (n=36) | both objects named → underspecified |
| **+33.4** | "put the small green cube on the yellow cube" | 55.6% (n=36) | "put the teal-green cube on the yellow cube" | 22.2% (n=36) | plain color → compound color · **PURE ✂ single edit** |
| **+8.4** | "put the green cube on the yellow cube" | 15.3% (n=72) | "stack the green block on the yellow block" (nominal) | 6.9% (n=72) | cube → block |

### 3. Carrot → keyboard — real modest rescue (+15.2 confirmed)
![](results/search/scenes/carrot_on_keyboard.png)

**Confirmed (n=72):** "set the carrot on the black keyboard" **31.9%** vs
nominal "put carrot on keyboard" **16.7%** → **+15.2pp**.

**Scene reading:** slim black keyboard, low contrast, slightly raised —
balancing a rigid carrot on a narrow hard deck is mechanically hard; the
ceiling (~32) is visible in the picture.

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+27.8** | "place the carrot on the keyboard keys" | 36.1% (n=36) | same phrase, held-out | 8.3% (n=72) | winner's-curse collapse |
| **+22.2** | "place the carrot on the black keyboard" | 25.0% (n=36) | "put the carrot on the keys" | 2.8% (n=36) | whole-name → part-name |
| **+8.3** | "place the carrot on the black keyboard" | 25.0% (n=36) | "put carrot on keyboard" (nominal) | 16.7% (n=36) | +black, place-frame |
| **+4.1** | "set the carrot on the black keyboard" | 31.9% (n=72) | "place the carrot on the black keyboard" | 27.8% (n=72) | set → place · **PURE ✂ single edit** |

### 4. Carrot → wheel — ceiling-limited (confirmation pending)
![](results/search/scenes/carrot_on_wheel.png)

**Best so far:** "put the carrot on the black wheel" 38.9% (n=36) vs nominal
"put carrot on wheel" 22.2% (n=36); held-out confirmation in flight.

**Scene reading:** a small tire lying flat — narrow curved ring, shallow hub;
mechanically awkward for a long carrot. Low ceiling visible.

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+33.3** | "put the carrot on the black wheel" | 38.9% (n=36) | "put the carrot on the rim" | 5.6% (n=36) | whole-name → part-name · **PURE ✂ single edit** |
| **+22.3** | "place the carrot on the black tire" | 30.6% (n=36) | "put the carrot on the tire" | 8.3% (n=36) | +black, place-frame |
| **+13.9** | "place the carrot on the black tire" | 30.6% (n=36) | "place the carrot on the black rubber tire" | 16.7% (n=36) | +material adjective · **PURE ✂ single edit** |

### 5. Coke can → plate — nominal unbeaten (confirmed tie)
![](results/search/scenes/coke_can_on_plate.png)

**Confirmed (n=72):** "put the coke on the plate" 84.7% vs nominal "put coke
can on plate" 83.3% — **tie**; 48 candidates found nothing better.

**Scene reading:** the plate is pale yellow-green — ambiguously "yellow" or
"green", which likely explains plate-adjective instability across tasks.

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+30.5** | "put the coke can on the plate" | 61.1% (n=36) | "put the can on the plate" | 30.6% (n=36) | drop "coke" entirely · **PURE ✂ single edit** |
| **+30.5** | "put the coke can on the plate" | 61.1% (n=36) | "put the coke can on the dish" | 30.6% (n=36) | plate → dish · **PURE ✂ single edit** |
| **+25.0** | "pick up the coke can and place it on the plate" | 55.6% (n=36) | "take the coke can and put it on the plate" | 30.6% (n=36) | pick-up → take |
| **+22.2** | "put the coke can on the plate" | 61.1% (n=36) | "put the coke can on top of the plate" | 38.9% (n=36) | on → on top of · **PURE ✂ single edit** |
| **+19.4** | "put the coke can on the plate" | 61.1% (n=36) | "put the coke can on the yellow plate" | 41.7% (n=36) | +yellow · **PURE ✂ single edit** |
| **+16.7** | "put the coke can on the plate" | 61.1% (n=36) | "put the cola can on the plate" | 44.4% (n=36) | coke → cola · **PURE ✂ single edit** |
| **+13.9** | "put the coke can on the plate" | 61.1% (n=36) | "put the soda can on the plate" | 47.2% (n=36) | coke → soda · **PURE ✂ single edit** |
| **+4.2** | "set the coke can on the plate" | 65.3% (n=72) | "put coke can on plate" (nominal) | 61.1% (n=36) | set ≈ put (within noise) |
| **+2.8** | "put the coke can on the plate" | 61.1% (n=36) | "put the pepsi can on the plate" | 58.3% (n=36) | coke → pepsi (wrong brand ≈ free) · **PURE ✂ single edit** |
| **+2.8** | "put the coke on the plate" | 63.9% (n=36) | "put the coke can on the plate" | 61.1% (n=36) | dropping "can" costs nothing · **PURE ✂ single edit** |
| **±0.0** | "please put the coke can on the plate" | 61.1% (n=36) | "put the coke can on the plate" | 61.1% (n=36) | "please" is free · **PURE ✂ single edit** |

### 6. Carrot → plate — nominal WINS (confirmed)
![](results/search/scenes/carrot_on_plate.png)

**Confirmed (n=72):** nominal "put carrot on plate" **48.6%** beats
search-best "place the orange carrot on the green plate" 45.8%.

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+30.5** | "put the orange carrot on top of the plate" | 61.1% (n=36) | same phrase, held-out | 30.6% (n=72) | sharpest winner's-curse collapse |
| **+16.7** | "place the orange carrot on the green plate" | 55.6% (n=36) | "put the carrot on the plate" | 38.9% (n=36) | +colors (search split only) |
| **+2.8** | "put carrot on plate" (nominal) | 48.6% (n=72) | "place the orange carrot on the green plate" | 45.8% (n=72) | color gain did NOT survive |
| **±0.0** | "put the carrot on the plate" | 38.9% (n=36) | "put the carrot on the dish" | 38.9% (n=36) | dish HARMLESS here (cf. scene 5) · **PURE ✂ single edit** |

### 7. Spoon → towel — nominal WINS (confirmed)
![](results/search/scenes/spoon_on_towel.png)

**Confirmed (n=72):** nominal "put the spoon on the towel" **69.4%** beats
"set the spoon down on the towel" 62.5%.

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+19.4** | "put the spoon on top of the towel" | 58.3% (n=36) | "put the spoon on the blue towel" | 38.9% (n=36) | +blue — despite "blue towel" ×5,815 in training text |
| **+16.6** | "put the spoon on top of the towel" | 58.3% (n=36) | "put the spoon on the cloth" | 41.7% (n=36) | towel → cloth |
| **+9.7** | "set the spoon down on the towel" | 72.2% (n=36) | same phrase, held-out | 62.5% (n=72) | leader regression |

### 8. Eggplant → basket — null control (confirmed plateau)
![](results/search/scenes/put_eggplant_in_basket.png)

**Confirmed (n=72):** plateau 94.4–95.8%; nominal within noise of everything.

**Scene reading:** the "yellow basket" is literally a **yellow dish rack in a
toy sink** — solving the legacy mystery cell ("aubergine…dish rack" 92.7%).
Huge open target → nothing for phrasing to fix.

| Δ (pp) | better phrase | success | worse phrase | success | feature |
|---|---|---|---|---|---|
| **+38.9** | "put the eggplant in the yellow basket" | 97.2% (n=36) | "put the eggplant in the yellow bin" | 58.3% (n=36) | basket → bin · **PURE ✂ single edit** |
| **+36.1** | "put the eggplant in the yellow basket" | 97.2% (n=36) | "eggplant in yellow basket" | 61.1% (n=36) | full sentence → telegram |


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

*Wheel confirmation pending; its section updates on landing.*