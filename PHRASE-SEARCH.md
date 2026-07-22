# PHRASE-SEARCH — LLM-guided rollout search over instruction phrasings

**Method.** Per task: rounds of 16 phrases proposed by Claude (informed by all
prior boards), each rolled on the SEARCH split (layouts 0–17, ×2, n=36/phrase,
SE ≈ 8pp); survivors CONFIRMED on the held-out split (layouts 18–23, ×12,
n=72). Cross-split levels differ by up to ~20pp — only within-split contrasts
are valid. Single-board deltas under ~16pp are noise-range.

**Stability flags — exact meaning.**
- **[1] single measurement**: each phrase in the pair was executed in exactly
  ONE search round = 18 distinct object layouts (episode_ids 0–17) × 2
  repeats = **36 sim episodes per phrase** (72 for the pair). Repeats within a
  layout share object placement and differ only in the policy's stochastic
  decode, so the effective sample is closer to 18 independent clusters than
  36 episodes; per-phrase SE ≈ 8pp at mid success rates, SE of the pair's
  Δ ≈ 11pp. A [1] delta under ~16pp is noise-range; even a large [1] delta
  is provisional until replicated.
- **[R] replicated**: the same pair (or the same contrast in a near-identical
  frame) was re-measured in ≥2 independent rounds — a fresh 36-episode board
  each time — with consistent sign. Evidence grade: rule-worthy.
- **[U] unstable**: re-measured and the sign or magnitude flipped across
  rounds/frames. Observed, not actionable.
- **Confirmed** (headline table): measured on the HELD-OUT split, 6 layouts
  (episode_ids 18–23) × 12 repeats = **72 episodes per phrase**, layouts never
  touched during search — the only numbers quoted as results.

**Confirmed headlines (held-out split, n=72).**
| task | best found | nominal (same split) | headroom |
|---|---|---|---|
| coke-ramekin | "place the red coke can inside the white bowl" **95.8** | 36.1 | **+59.7** |
| coke-plate | "put the coke on the plate" 84.7 | 83.3 | tie |
| eggplant-basket (1 round, search split) | plateau 97.2 | ~97 | none |

**The headroom law:** better-than-nominal headroom is inversely proportional
to nominal quality. Search pays where the given instruction is bad.

---

## Minimal pairs with drastic effects

Search-split numbers (n=36) unless noted. Δ = worse-minus-better.

### Receptacle noun (the biggest lever)
| pair (same frame) | values | Δ | note |
|---|---|---|---|
| ramekin: "white **bowl**" vs "white **dish**" | 63.9 vs 2.8 | **−61** | [1] the most violent single-token swap recorded |
| eggplant: "yellow **basket**" vs "yellow **bin**" | 97.2 vs 58.3 | **−39** | [1] near-synonyms are not synonyms to π0 |
| coke-plate: "the **plate**" vs "the **dish**" | 61.1 vs 30.6 | **−31** | [1] |
| carrot-plate: "the plate" vs "the dish" | 38.9 vs 38.9 | 0 | [1] …yet dish is HARMLESS here — toxicity is task-contextual |
| ramekin: "white bowl" vs "white **cup**" | 61.1 vs 44.4 | −17 | [R] two rounds |
| ramekin: "white bowl" vs "white **ramekin**" | 63.9 vs 22.2 | −42 | [1] the object's true name loses to its look-alike name |

### Source-object token identity
| pair | values | Δ | note |
|---|---|---|---|
| "the **coke** can" vs "the **soda** can" | 61.1 vs 47.2 | −14 | [R] also ramekin-side 27.8 for soda forms |
| "the coke can" vs "the **cola** can" | 61.1 vs 44.4 | −17 | [R] |
| "the coke can" vs "the (bare) can" | 61.1 vs 30.6 | **−31** | [1] |
| "the coke can" vs "the **pepsi** can" | 61.1 vs 58.3 | −3 | [1] wrong BRAND ≈ free; wrong CATEGORY ≈ fatal |
| "the coke" vs "the coke can" | ~+4 to +8 for dropping "can" | — | [R] six r3 forms ≥63.9 |

### Adjectives — value is per-object, not universal
| pair | values | Δ | note |
|---|---|---|---|
| ramekin: "**white** bowl" vs "**ceramic** bowl" | 63–72 vs 27.8 | **~−35** | [1] COLOR attributes ≫ MATERIAL attributes |
| coke-plate: "the plate" vs "the **yellow** plate" | 61.1 vs 41.7 | −19 | [R] adjective on familiar receptacle = dead weight |
| carrot-plate: bare vs "**orange** carrot…**green** plate" | 38.9 vs 55.6 | **+17** | [1] same construction HELPS here |
| spoon: "the towel" vs "the **blue** towel" | ~58 vs 38.9 | −19 | [1] |

### Prepositions and relations
| pair | values | Δ | note |
|---|---|---|---|
| coke-plate: "on" vs "**onto**" | 61.1 vs 44.4 (r1); 63.9 vs 66.7 (r3, "coke" frame) | −17 / +3 | **[U]** unstable — do not rule on it |
| coke-plate: "on" vs "**on top of**" | 61.1 vs 38.9 | −22 | [1] yet "on top of" was FINE on spoon (58.3) and carrot-plate (52.8) |
| ramekin: "place it **inside**" vs "put it **in**" | 61.1 vs 44.4 | −17 | [1] verb+prep bundle |
| ramekin: nominal "**on** ramekin" vs "in/inside bowl" forms | 0–11 vs 60–72 | **~−55** | [R] wrong relation + wrong noun = the task's whole failure |

### Verbs and clause structure
| pair | values | Δ | note |
|---|---|---|---|
| "set" vs "put" (matched frames) | ≈ equal (63.9 vs 61.1; 66.7 vs 63.9 ×3 boards) | ~0 | [R] "set" was WRONGLY banned by rules-v1 (its bad rap came from a confounded pair) |
| "please put…" vs "put…" | 61.1 vs 61.1 | 0 | [1] politeness free — another wrong ban |
| coke-plate: "**take** X and put" vs "**pick up** X and place" | 30.6 vs 55.6 | −25 | [1] first-clause verb matters |
| ramekin: "**move**…into" vs "place…inside" | 22.2 vs 61.1 | −39 | [1] |
| ramekin: single-clause "place the…" vs "pick up the… and place it…" | 69.4 vs 61.1 | +8 | [R] single-clause ≥ two-clause |
| ramekin: ±"upright" (same frame) | 61.1 vs 58.3 (r2); 72.2 vs 55.6 (r1) | pooled ~+10 for dropping | [U→R-ish] orientation word unnecessary, mildly negative |

### Register and micro-features
| pair | values | Δ | note |
|---|---|---|---|
| trailing period: "…on the plate" vs "…on the plate**.**" | 61.1 vs 47.2 | −14 | [U] cross-round measurement — flag, don't rule |
| telegram: "put the eggplant in the yellow basket" vs "eggplant in yellow basket" | 97.2 vs 61.1 | **−36** | [1] too terse breaks it; yet "put coke can on plate" (terse-ish) is optimal there |
| spoon: "put…down on" vs "put…on" | 63.9 vs mid-band | +~6 | [1] particle "down" mildly helps spoon |

### Additions from the six-task extension (round 1–2 boards, 2026-07-22)

**Ceiling-limited tasks (the law's refinement).** Keyboard: best "place the
carrot on the black keyboard" 25.0 vs nominal 16.7 (+8.3). Wheel: "place the
carrot on the black tire" 30.6 vs nominal 22.2 (+8.4). Both bad-nominal tasks,
both nearly headroom-free — their phrasing-conditional CEILINGS are low.
Final law: **headroom = ceiling − nominal**; big rescues need a bad nominal
AND a capable policy.

| new pair | values | Δ | note |
|---|---|---|---|
| stack: "…green **cube** on yellow **cube**" family vs "…**block**" family | 44–58 vs 22–33 | **~+25** | [R] the cube>block effect at full size; nominal ("stack the green block…") 22.2 |
| stack: telegram "green cube on yellow cube" vs nominal | 58.3 vs 22.2 | **+36** | [1] verbless wins on stack — yet telegram LOST 36 on eggplant; terseness is task-specific |
| stack: "**teal-green** cube" vs "green cube" | 22.2 vs 44–58 | **~−30** | [1] compound/unusual color adjectives are corpus-alien poison (rules-v1 emitted this) |
| stack: "stack the cubes" (underspecified) | 0.0 | — | [1] both objects must be named |
| wheel: "black tire" vs "black **rubber** tire" | 30.6 vs 16.7 | −14 | [1] color>material, third task replication |
| keyboard: "the keys" vs "the keyboard" | 2.8 vs ~17–25 | ~−17 | [1] part-name loses to whole-name |
| carrot-plate: "orange carrot…green plate" vs bare | 55.6/58.3 vs 38.9 | **+17** | [R] color adjectives HELP here (hurt on coke-plate/spoon) — per-object, not global |
| spoon: "put the spoon **down** on" vs plain forms | 63.9 vs ~55–58 | +6 | [1] particle mildly helps |
| spoon: "towel" vs "**cloth**" | ~58 vs 41.7 | −17 | [1] another near-synonym cliff |
| eggplant: whole plateau at 97.2 | — | — | [R] null control confirmed: good nominals leave nothing to find |

---

## Reading discipline
1. Per-phrase SE ≈ 8pp at n=36: singles under ~16pp are suggestive only.
2. Winner's curse: every board's top regresses on replication (75.0→"set" plateau;
   72.2→61.1 ramekin leader) — trust replicated bands, not single leaders.
3. Cross-task transfer of micro-rules is POOR (dish, on-top-of, adjectives all
   flip sign between tasks). What transfers: name objects with the right
   concrete token, use the right relation for the geometry, stay in plain
   imperative register, and prefer the task's nominal when it is already good.

*Living document — boards for keyboard, wheel, stack, and second rounds still
landing; confirmed numbers supersede search-split numbers where present.*
