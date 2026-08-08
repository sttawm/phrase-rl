# Previous rulebook: measured performance

Whole rulebook applied: estimated success 0.575 vs 0.593 for the unrephrased instruction (-0.018).

## Per-rule single-edit effects

Each rule applied ALONE to the same base instructions, so the
delta below is attributable to that rule and not to the rest of
the rulebook.

### rule_1
text: Rewrite the instruction as a short, plain imperative that keeps the same objects and goal.
delta vs base: -0.1334   (n=8 instructions)
    on search       inputs: -0.1334  (n=8)

## Sample rewrites from the whole rulebook

  [put stuffedduck in pan]
    in : 'place yellow duck in pot'
    out: '[dry-run:gemini] 7fce974fabc9'
  [turn lever vertical to front]
    in : 'pull the silver handle forward'
    out: '[dry-run:gemini] 621dc73ae4ca'
  [pick up pan from stove]
    in : 'pick up the small silver pot from the stove'
    out: '[dry-run:gemini] e2c13623aaa8'
  [put pot in sink]
    in : 'move the pot of fake food to the sink'
    out: '[dry-run:gemini] ec6229e4acec'
  [put blueberries on plate sink]
    in : 'place blue grapes on the yellow plate in the sink'
    out: '[dry-run:gemini] 736870842653'
  [fold the cloth from bottom right to top left]
    in : 'diagonally fold the green towel from bottom right to top left'
    out: '[dry-run:gemini] a0a0c38f7214'
  [move the silver pot to the upper right of table]
    in : 'slide the grey stuffed animal to the upper right'
    out: '[dry-run:gemini] 9af3b2893c5d'
  [fold the cloth from top to bottom]
    in : 'fold the blue cloth by pulling the top down'
    out: '[dry-run:gemini] 5aac1776ceba'
  [put detergent in sink]
    in : 'place the white bottle in the sink'
    out: '[dry-run:gemini] b6cc9e15bebb'
  [pick up any cup]
    in : 'pick up any cup from the rack'
    out: '[dry-run:gemini] 5252a3da701c'
  [wipe pot with sponge]
    in : 'use the sponge to wipe the pot'
    out: '[dry-run:gemini] a37748b5edd5'
  [unfold the cloth from bottom to top]
    in : 'flatten the orange rag from bottom to top'
    out: '[dry-run:gemini] fbd89ec11076'
  [put clothes in laundry machine]
    in : 'place the clothes into the open laundry machine'
    out: '[dry-run:gemini] 209ae9dfcb3c'
  [put carrot in pot or pan]
    in : 'put carrot in pot'
    out: '[dry-run:gemini] 09694e0708cf'
  [end effector reaching salmon]
    in : 'bring gripper close to the salmon'
    out: '[dry-run:gemini] fc8dfe765447'
  [put cup from anywhere into sink]
    in : 'put the blue plastic cup in the sink'
    out: '[dry-run:gemini] bae27d9c6ce5'
  [put brush into pot or pan]
    in : 'drop the brush inside the silver pot'
    out: '[dry-run:gemini] effaf077e38b'
  [put carrot on plate]
    in : 'place carrot onto the green plate'
    out: '[dry-run:gemini] e1de4728a972'
  [put bowl on plate]
    in : 'place the bowl onto the plate'
    out: '[dry-run:gemini] b4c0eb2cf055'
  [pick up glue and put into drawer]
    in : 'put the red tube of glue in the top left organizer'
    out: '[dry-run:gemini] aa9ab54d088e'
  [pick up blue pen and put into drawer]
    in : 'move the blue pen to the top left plastic drawer'
    out: '[dry-run:gemini] e02e89970779'
  [pick up scissors and put into drawer]
    in : 'pick up scissors and place in drawer'
    out: '[dry-run:gemini] aec545de50da'
  [put pepper in pot or pan]
    in : 'place the pepper in the pot'
    out: '[dry-run:gemini] affa8ef68d95'
  [move light switch to the right]
    in : 'toggle the switch right'
    out: '[dry-run:gemini] 14fa3c1c07c5'