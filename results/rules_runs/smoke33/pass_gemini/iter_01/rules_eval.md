# Previous rulebook: measured performance

Whole rulebook applied: estimated success 0.545 vs 0.593 for the unrephrased instruction (-0.048).

## Per-rule single-edit effects

Each rule applied ALONE to the same base instructions, so the
delta below is attributable to that rule and not to the rest of
the rulebook.

### rule_1
text: Keep the original wording unless a rule below applies.
delta vs base: -0.0092   (n=8 instructions)
    on search       inputs: -0.0092  (n=8)

### rule_2
text: Use common household nouns.
delta vs base: -0.0810   (n=8 instructions)
    on search       inputs: -0.0810  (n=8)

### rule_3
text: Keep existing color adjectives.
delta vs base: -0.1177   (n=8 instructions)
    on search       inputs: -0.1177  (n=8)

## Sample rewrites from the whole rulebook

  [put stuffedduck in pan]
    in : 'place yellow duck in pot'
    out: '[dry-run:gemini] 610341445ca0'
  [turn lever vertical to front]
    in : 'pull the silver handle forward'
    out: '[dry-run:gemini] c5513e1d7816'
  [pick up pan from stove]
    in : 'pick up the small silver pot from the stove'
    out: '[dry-run:gemini] c57e038be017'
  [put pot in sink]
    in : 'move the pot of fake food to the sink'
    out: '[dry-run:gemini] 329c169b834d'
  [put blueberries on plate sink]
    in : 'place blue grapes on the yellow plate in the sink'
    out: '[dry-run:gemini] a9e35c240f86'
  [fold the cloth from bottom right to top left]
    in : 'diagonally fold the green towel from bottom right to top left'
    out: '[dry-run:gemini] 6af0c68a9921'
  [move the silver pot to the upper right of table]
    in : 'slide the grey stuffed animal to the upper right'
    out: '[dry-run:gemini] 382e62c1eecf'
  [fold the cloth from top to bottom]
    in : 'fold the blue cloth by pulling the top down'
    out: '[dry-run:gemini] 21b10f76c73b'
  [put detergent in sink]
    in : 'place the white bottle in the sink'
    out: '[dry-run:gemini] 37dd8ad2a353'
  [pick up any cup]
    in : 'pick up any cup from the rack'
    out: '[dry-run:gemini] 1ff946efa80e'
  [wipe pot with sponge]
    in : 'use the sponge to wipe the pot'
    out: '[dry-run:gemini] 1dd3d488370b'
  [unfold the cloth from bottom to top]
    in : 'flatten the orange rag from bottom to top'
    out: '[dry-run:gemini] ca6459f4683c'
  [put clothes in laundry machine]
    in : 'place the clothes into the open laundry machine'
    out: '[dry-run:gemini] 22a500ce28f7'
  [put carrot in pot or pan]
    in : 'put carrot in pot'
    out: '[dry-run:gemini] 06557b96b4ac'
  [end effector reaching salmon]
    in : 'bring gripper close to the salmon'
    out: '[dry-run:gemini] 2494cf975c88'
  [put cup from anywhere into sink]
    in : 'put the blue plastic cup in the sink'
    out: '[dry-run:gemini] 940a6c78a6aa'
  [put brush into pot or pan]
    in : 'drop the brush inside the silver pot'
    out: '[dry-run:gemini] 48abeee6e8e4'
  [put carrot on plate]
    in : 'place carrot onto the green plate'
    out: '[dry-run:gemini] cfe675dca31f'
  [put bowl on plate]
    in : 'place the bowl onto the plate'
    out: '[dry-run:gemini] 6ced14f2b52d'
  [pick up glue and put into drawer]
    in : 'put the red tube of glue in the top left organizer'
    out: '[dry-run:gemini] 7f68f051eab3'
  [pick up blue pen and put into drawer]
    in : 'move the blue pen to the top left plastic drawer'
    out: '[dry-run:gemini] 10c5100126c2'
  [pick up scissors and put into drawer]
    in : 'pick up scissors and place in drawer'
    out: '[dry-run:gemini] 6fddba7ef480'
  [put pepper in pot or pan]
    in : 'place the pepper in the pot'
    out: '[dry-run:gemini] c51fe2759398'
  [move light switch to the right]
    in : 'toggle the switch right'
    out: '[dry-run:gemini] 0e24edf9ce62'