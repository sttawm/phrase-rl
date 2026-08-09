===RULE NOTES===
1. **Adherence**: Near perfect. The rewrite samples show strict compliance with the lowercase constraint and the complete removal of periods and capitalization. 
**Performance**: +2.4% overall success rate (N=1450). The gains are highly consistent across all input kinds (original: +2.2%, natural: +2.5%, adversarial: +2.3%, rephrased: +2.6%). 
**Action**: Keep this rule exactly as is. It provides a foundational normalization layer that the robot policy clearly relies on, and the applier executes it flawlessly without interfering with other rules.

2. **Adherence**: High, but it occasionally creates friction when interacting with Rule 5 (spatial relations). Because the applier is forced to use a "highly frequent action verb," it sometimes translates sliding or pushing actions awkwardly into "move" or "put," losing the specific contact dynamics of the original prompt.
**Performance**: +6.1% overall (N=1450). It performs exceptionally well on adversarial (+8.4%) and rephrased (+7.2%) inputs by grounding chaotic prompts back to a known action space.
**Action**: Keep the rule, but expand the allowed verb list to capture a slightly wider range of physical interactions. Change the phrasing to: `Begin the instruction with a highly frequent action verb, such as "move", "put", "place", "pick up", "take", "remove", "fold", "unfold", "push", or "pull".`

3. **Adherence**: Moderate. The applier reliably adds "the" before the primary object being manipulated, but frequently misses secondary objects or receptacles (e.g., outputting "put the red block in bowl"). 
**Performance**: +0.8% overall (N=1450). However, this masks a sharp conditional split: it improves natural inputs (+3.5%) but actively degrades adversarial inputs (-2.2%). Adding "the" before nonsense or out-of-distribution nouns in adversarial prompts seems to create phrasing the policy has never seen.
**Action**: Narrow the rule so it works in tandem with Rule 4, rather than applying blindly to all nouns. Change the wording to: `Include the definite article "the" before recognized objects and receptacles (e.g., write "put the cup on the plate"). Do not add "the" before abstract or unrecognized words.`

4. **Adherence**: Excellent. The applier is highly successful at mapping complex, out-of-distribution descriptions (e.g., "crimson chalice") down to the approved basic terms ("red bowl" or "red cup"). 
**Performance**: +8.5% overall (N=1450). This is the strongest rule in the book, showing massive gains on rephrased (+9.8%) and adversarial (+12.1%) inputs where vocabulary normalization is most critical.
**Action**: Keep the rule, but add transparency to the materials list. The samples show that when the original prompt mentions "clear" or "see-through," the applier drops the attribute entirely because it isn't in the approved list. Insert `"clear"` into the list of simple colors/materials.

5. **Adherence**: Good, though it sometimes results in preposition stacking when the applier tries to preserve too much of the original prompt's spatial complexity (e.g., "out of the top of").
**Performance**: +3.2% overall (N=1450). It shows steady improvements across natural (+3.8%) and original (+3.0%) inputs.
**Action**: Simplify the rule to enforce brevity and prevent it from pulling against the simplicity mandated by Rule 4. Add a limiting clause at the end: `Use only one spatial preposition per object relationship; do not stack them (e.g., use "in" instead of "down into").`

6. **Adherence**: Mixed. The applier easily strips politeness markers ("please"), exact measurements, and brand names. However, it struggles significantly with conditionals ("if"). Instead of removing the conditional logic, the applier often just rewrites the "if" clause using simpler words, violating the spirit of the rule.
**Performance**: +4.7% overall (N=1450) *when obeyed*. The removal of adverbs of manner (+5.5% on natural inputs) pairs perfectly with Rule 2, leaving a clean, verb-led command.
**Action**: Split this rule. The negative constraints on politeness, adverbs, measurements, and brands are working well and should remain as Rule 6. Conditionals need to be dropped from this list and given their own dedicated rule that tells the applier *how* to resolve them. Create a new Rule 7: `If the input contains a conditional or "if" clause, remove the condition entirely and rewrite the instruction as a direct command assuming the condition is true.`

===SUGGESTIONS===
"push the [color] [noun] to the [location]" vs "move the [color] [noun] to the [location]" — Tells us if specific contact verbs (push) outperform generic translation verbs (move) for sliding tasks.
"[verb] [noun] [preposition] [receptacle]" (dropping "the" entirely) — Tells us if Rule 3 is actually necessary for policy success, or if a purely minimalist, telegraphic baseline is better.
"[verb] the [color] [noun]" vs "[verb] the [material] [noun]" — Tells us which attribute (color vs. material) the policy prioritizes for disambiguation when an object has both, helping us order the adjectives in Rule 4.
"[verb] the [noun] [spatial relation]" (e.g., "move the block left") vs "[verb] the [noun] to the [spatial relation]" (e.g., "move the block to the left") — Tells us if the policy requires prepositional anchors for directional commands, which would refine Rule 5.