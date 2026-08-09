# Previous rulebook: measured performance

Whole rulebook applied: estimated success 0.575 vs 0.728 for the unrephrased instruction (-0.153).

## Per-rule single-edit effects

Each rule applied ALONE to the same base instructions, so the
delta below is attributable to that rule and not to the rest of
the rulebook.

### rule_1
text: Format the output strictly in lowercase letters with no punctuation. Never capitalize the first letter and never use a period.
delta vs base: +0.0000   (n=3 instructions)
    on search       inputs: +0.0000  (n=3)

### rule_2
text: Begin the instruction with a highly frequent action verb, such as "move", "put", "place", "pick up", "take", "remove", "fold", or "unfold".
delta vs base: +0.1016   (n=3 instructions)
    on search       inputs: +0.1016  (n=3)

### rule_3
text: Include the definite article "the" before nouns (e.g., write "put the cup on the plate" instead of "put cup on plate").
delta vs base: +0.0000   (n=3 instructions)
    on search       inputs: +0.0000  (n=3)

### rule_4
text: Describe objects using basic category nouns (e.g., "pot", "bowl", "block", "cloth"), simple colors (e.g., "blue", "yellow", "red", "green", "orange", "white", "grey", "purple"), and common materials (e.g., "silver", "metal", "plastic", "wooden").
delta vs base: -0.4321   (n=3 instructions)
    on search       inputs: -0.4321  (n=3)

### rule_5
text: Specify locations and spatial relations using common prepositions and positional words (e.g., "on", "in", "from", "to", "into", "out of", "left", "right", "top", "bottom", "middle", "corner").
delta vs base: -0.1803   (n=3 instructions)
    on search       inputs: -0.1803  (n=3)

### rule_6
text: Do not use politeness markers (e.g., "please"), adverbs of manner (e.g., "carefully"), conditionals (e.g., "if"), exact measurements, or brand names.
delta vs base: +0.0000   (n=3 instructions)
    on search       inputs: +0.0000  (n=3)

## Sample rewrites from the whole rulebook

  [put pan on stove from sink]
    in : 'place the pan on the blue burner'
    out: 'place the pan on the blue burner'
  [pick up glass cup]
    in : 'pick up glass cup'
    out: 'pick up the glass cup'
  [put cup on plate]
    in : 'set the orange cup on the plate'
    out: 'put the orange cup on the plate'
  [put fork from basket to tray]
    in : 'move the fork from the basket to the tray'
    out: 'move the silver metal fork from the grey plastic basket on the left to the black tray on the right'
  [put pear in bowl]
    in : 'drop pear in blue bowl'
    out: 'put the pear in the blue bowl'
  [put cup on plate]
    in : 'set the orange cup onto the plate'
    out: 'put the orange cup on the plate'