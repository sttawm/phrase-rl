# Mechanism case families (all numbers = measured rollout success, greedy ×12 unless noted)

## Lexeme preference within concrete nouns (the cube/block pair)
Task stack_cube, GT says "block". π0 prefers "cube": +12pp measured on a powered
pair. Invisible to every offline reward feature (the one universal miss at the
exam's 67/68 ceiling). Both words are concrete and in-corpus; frequency/embodiment
in Bridge appears to decide.

## OOV receptacle: ramekin family (coke_can_on_ramekin_clean)
- v6's "Pick up the red cola can and place it upright inside the white bowl." → 64.2%  (resolve to in-vocab NEIGHBOR)
- frozen echo of full circumlocution ("hollow white ceramic cup-like container") → 42.0%  (π0 parses this one)
- frozen_selftrace rephrase → 46.2%
- SFT-v1 "…in the white object" → 3.8%  (vague category placeholder = destruction)
- SFT-v2 "…in the white object that looks like a cup" → 24.7%  (partial)
Trace evidence: the self-trace referent map literally offers "→ ceramic bowl"; v2's
greedy declined it. Sampled: "white bowl" draws hit 54.2% (winner's-curse caveat).

## OOV receptacle: keyboard family (carrot_on_keyboard_clean)
- v6's "…place it on the black keyboard." → 22.9%  (KEEP the OOV noun)
- SFT-v2 "put the orange carrot on top of the keyboard" → 16.7%
- frozen echo of circumlocution ("black peripheral device used for typing") → 0.3%  (π0 does NOT parse this one)
- SFT-v1 "…above the mouse pad" → 3.1%  (wrong in-vocab neighbor = worst kind of miss)
Contrast with ramekin: π0's parsing of rich descriptions is OBJECT-DEPENDENT.

## Wheel family (carrot_on_wheel_clean) — hard task, small numbers
- SFT-v2 "pick the orange carrot and put it on the wheel" → 15.6% (task record)
- SFT-v1 same shape → 10.4%; frozen "tire" repair → 8.7%; v6 → 3.1%

## Spoon: register sensitivity (spoon_on_towel)
- SFT-v2 "put the spoon on top of the towel" → 63.9%
- SFT-v1 "put the spoon onto the towel" → 59.7%
- frozen_bare "Place the spoon in the center of the towel." → 40.6%
- frozen_selftrace → 34.0%; v6's phrase → 25.0%
Small canonical-register differences move this task a lot.

## Deixis (coke_can_on_plate_clean)
- GT-style "put coke can on plate" → 64.6% (n=864, from the pairs file)
- frozen_selftrace version → 29.5%; frozen_gemini → 27.4%
- SFT-v2 "pick the can on the left side and put it on the yellow plate on the right side" → 10.1%
- SFT-v1 near-identical deixis-preserving form → 5.9%; echo of full hostile → 1.0%
Preserved left/right deixis in long sentences does NOT rescue this task; the short
canonical form dominates. Length/complexity itself may be the enemy here.

## Benign-input headroom (eggplant_in_basket)
- canonical "put the eggplant in the yellow basket" → 94.4% (SFT-v1 and v2 tie)
- frozen echo of mildly-hostile "Arrange the eggplant neatly in the yellow bin." → 53.8%
Canonicalizing even MILD hostility is worth +40pp when the target is in-vocab.
