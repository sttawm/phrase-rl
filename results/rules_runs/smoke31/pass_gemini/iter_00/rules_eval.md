# Previous rulebook: measured performance

Whole rulebook applied: estimated success 0.474 vs 0.593 for the unrephrased instruction (-0.119).

## Per-rule single-edit effects

Each rule applied ALONE to the same base instructions, so the
delta below is attributable to that rule and not to the rest of
the rulebook.

### rule_1
text: Rewrite the instruction as a short, plain imperative that keeps the same objects and goal.
delta vs base: -0.0211   (n=8 instructions)
    on search       inputs: -0.0211  (n=8)

## Sample rewrites from the whole rulebook

  [put stuffedduck in pan]
    in : 'place yellow duck in pot'
    out: '[dry-run:gemini] 171464f3b30a'
  [turn lever vertical to front]
    in : 'pull the silver handle forward'
    out: '[dry-run:gemini] b1f84e140a99'
  [pick up pan from stove]
    in : 'pick up the small silver pot from the stove'
    out: '[dry-run:gemini] 195de983d7e3'
  [put pot in sink]
    in : 'move the pot of fake food to the sink'
    out: '[dry-run:gemini] fa8bb4efc7d3'
  [put blueberries on plate sink]
    in : 'place blue grapes on the yellow plate in the sink'
    out: '[dry-run:gemini] 1d826ba9ac9b'
  [fold the cloth from bottom right to top left]
    in : 'diagonally fold the green towel from bottom right to top left'
    out: '[dry-run:gemini] d3264758e504'
  [move the silver pot to the upper right of table]
    in : 'slide the grey stuffed animal to the upper right'
    out: '[dry-run:gemini] cca6993d0569'
  [fold the cloth from top to bottom]
    in : 'fold the blue cloth by pulling the top down'
    out: '[dry-run:gemini] 5d38986acda5'
  [put detergent in sink]
    in : 'place the white bottle in the sink'
    out: '[dry-run:gemini] f1742c0ef0a5'
  [pick up any cup]
    in : 'pick up any cup from the rack'
    out: '[dry-run:gemini] 4414cafba3e7'
  [wipe pot with sponge]
    in : 'use the sponge to wipe the pot'
    out: '[dry-run:gemini] e08c3d040195'
  [unfold the cloth from bottom to top]
    in : 'flatten the orange rag from bottom to top'
    out: '[dry-run:gemini] edefd6552979'
  [put clothes in laundry machine]
    in : 'place the clothes into the open laundry machine'
    out: '[dry-run:gemini] 02e005835878'
  [put carrot in pot or pan]
    in : 'put carrot in pot'
    out: '[dry-run:gemini] 2e723069e6ee'
  [end effector reaching salmon]
    in : 'bring gripper close to the salmon'
    out: '[dry-run:gemini] c8f211e83298'
  [put cup from anywhere into sink]
    in : 'put the blue plastic cup in the sink'
    out: '[dry-run:gemini] 4cf28c445f2e'
  [put brush into pot or pan]
    in : 'drop the brush inside the silver pot'
    out: '[dry-run:gemini] d383a23d8750'
  [put carrot on plate]
    in : 'place carrot onto the green plate'
    out: '[dry-run:gemini] cba2aa39e102'
  [put bowl on plate]
    in : 'place the bowl onto the plate'
    out: '[dry-run:gemini] 239f1c37e48c'
  [pick up glue and put into drawer]
    in : 'put the red tube of glue in the top left organizer'
    out: '[dry-run:gemini] 9b2e4a29f257'
  [pick up blue pen and put into drawer]
    in : 'move the blue pen to the top left plastic drawer'
    out: '[dry-run:gemini] f9cebdb920e5'
  [pick up scissors and put into drawer]
    in : 'pick up scissors and place in drawer'
    out: '[dry-run:gemini] 57082971d079'
  [put pepper in pot or pan]
    in : 'place the pepper in the pot'
    out: '[dry-run:gemini] a33a066885cd'
  [move light switch to the right]
    in : 'toggle the switch right'
    out: '[dry-run:gemini] 54a1975bd7e6'