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

