# Previous rulebook: measured performance

Whole rulebook applied: mean proxy logit 6.010 vs 6.408 for the unrephrased instruction (-0.399; higher is better, not a probability).

## Sample rewrites from the whole rulebook
(logit: higher is better; delta = rewrite - base, positive means
the rewrite beat leaving the instruction alone)

  [unfold the cloth from bottom to top]
    in : 'unfold orange cloth from bottom to top'
    out: 'Pull the yellow cloth from the bottom fold to the top.'
    logit: 3.614  (base 6.686, delta -3.072)
  [open oven]
    in : 'Swing the door to the oven open.'
    out: 'Pull the black handle on the grey toy oven door to swing it open.'
    logit: 7.115  (base 8.496, delta -1.381)
  [lift bowl]
    in : 'Provided that you have secured a sufficient grip on the hemispherical dish intended for liquids, proceed to increase its altitude relative to the current resting plane.'
    out: 'Grab the blue bowl and lift it up.'
    logit: 6.298  (base 4.897, delta +1.401)
  [put eggplant on plate]
    in : 'put the eggplant on the green plate'
    out: 'Place the purple toy eggplant on the light-green toy plate.'
    logit: 6.514  (base 6.774, delta -0.260)
  [end effector transition from object to object]
    in : 'Transfer the robot hand from the first piece to the second one.'
    out: 'Move the robot gripper from the orange cup to the silver colander.'
    logit: 2.705  (base 1.199, delta +1.506)
  [unfold the cloth from bottom right to top left]
    in : 'unfold the yellow cloth towards the top left'
    out: 'Spread the yellow cloth to the upper left.'
    logit: 5.638  (base 6.831, delta -1.193)
  [put spatula on plate sink]
    in : 'Regarding the handheld implement designed for lifting food, kindly transport said item to the uppermost surface of the circular eating vessel located at the sink.'
    out: 'Move the green-handled spatula to the top of the pink plate.'
    logit: 4.804  (base 1.665, delta +3.139)
  [put the blue figure on the top edge of the cloth]
    in : 'Place the blue figure on the upper edge of the cloth.'
    out: 'Put the blue figurine on the top border of the orange fabric.'
    logit: 4.287  (base 5.793, delta -1.506)
  [close oven]
    in : 'Shut the oven door.'
    out: 'Close the oven door.'
    logit: 8.321  (base 8.311, delta +0.011)
  [put pan in sink]
    in : 'Ascertain the whereabouts of the handled cooking vessel and subsequently ensure its physical relocation to the interior cavity of the dish-washing basin.'
    out: 'Find the silver pot and place it in the sink.'
    logit: 6.401  (base 4.620, delta +1.781)
  [pick up pot from sink]
    in : 'Lift the pot up out of the washbasin.'
    out: 'Pick up the silver pot from the sink.'
    logit: 5.631  (base 1.774, delta +3.857)
  [put sweet potato in pot which is in sink]
    in : "Grab the sweet potato and drop it in the pot that's resting in the sink."
    out: "Grab the red toy sweet potato and drop it in the silver pot that's resting in the sink."
    logit: 6.145  (base 6.736, delta -0.591)
  [put carrot on cutting board]
    in : 'Assuming you have correctly identified the orange, cone-shaped vegetable, kindly execute a spatial relocation protocol to position it squarely upon the cutting board.'
    out: 'Pick up the orange carrot and place it on the green cutting board.'
    logit: 7.091  (base 6.694, delta +0.396)
  [put sweet potato in pot]
    in : 'Transfer the sweet potato over to the pot.'
    out: 'Move the yellow toy potato into the silver pot.'
    logit: 3.509  (base 6.958, delta -3.449)
  [put pot in sink]
    in : 'The pot should be placed in the sink.'
    out: 'Put the silver pot in the sink basin.'
    logit: 7.310  (base 7.402, delta -0.092)
  [put fork from basket to tray]
    in : 'put fork from basket to tray'
    out: 'place silver fork from gray plastic cutlery basket to black and white utensil tray'
    logit: 5.353  (base 6.320, delta -0.967)
  [take spatula off plate sink]
    in : 'Please take the spatula off the plate and put it in the sink basin.'
    out: 'Pick up the green spatula from the pink plate and place it in the gray sink basin.'
    logit: 1.650  (base 7.581, delta -5.931)
  [put broccoli in pot]
    in : 'put the green broccoli piece inside the metal pot'
    out: 'place the light-green broccoli toy into the silver pot'
    logit: 6.278  (base 6.613, delta -0.335)
  [put carrot on cutting board]
    in : 'set the carrot on the cutting board'
    out: 'move the orange toy carrot onto the green cutting board'
    logit: 7.507  (base 7.836, delta -0.329)
  [zip zipper bag]
    in : 'Close the zipper bag.'
    out: 'Zip the black bag shut.'
    logit: 7.216  (base 7.654, delta -0.438)
  [open fridge]
    in : 'Get the refrigerator open for me.'
    out: 'Pull the handle of the white fridge open.'
    logit: 7.490  (base 7.552, delta -0.062)
  [put corn in pan which is on stove]
    in : 'In the pan on the stove, please put the corn.'
    out: 'Place the yellow plastic corn in the small silver pan on the stovetop.'
    logit: 5.037  (base 6.215, delta -1.177)
  [unfold the cloth from right to left]
    in : 'spread the orange cloth leftward'
    out: 'spread the yellow cloth to the left'
    logit: 6.131  (base 6.046, delta +0.085)
  [put banana in pot or pan]
    in : 'place banana in the silver pot'
    out: 'Put the yellow toy banana in the silver metal pot.'
    logit: 5.704  (base 6.297, delta -0.593)
  [put carrot on cutting board]
    in : 'Shift the carrot to the cutting board.'
    out: 'Move the orange toy carrot onto the light green toy cutting board.'
    logit: 7.119  (base 7.854, delta -0.735)
  [put pan in sink]
    in : 'put pan into the sink'
    out: 'place the pot in the basin'
    logit: 5.073  (base 7.467, delta -2.393)
  [close the drawer]
    in : 'Shut the drawer.'
    out: 'Close the red-faced drawer.'
    logit: 8.118  (base 8.650, delta -0.532)
  [fold the cloth from bottom left to top right]
    in : 'grab the bottom left corner of the orange towel and fold it top right'
    out: 'grasp the lower left corner of the blue microfiber cloth and fold it upper right'
    logit: 6.270  (base 6.568, delta -0.298)
  [close small4fbox flaps]
    in : 'Assuming the current assembly phase requires sealing, proceed to meticulously collapse the quartet of projecting leaves atop the miniature parcel.'
    out: 'Assuming the current step needs closing, go ahead to fold down the four open cardboard flaps on top of the small box.'
    logit: 7.666  (base 4.613, delta +3.053)
  [take the red object out of the pot and put it on the left burner]
    in : 'Targeting the red object that currently resides within the confines of the pot, please initiate a retrieval sequence to extract it, culminating in its placement atop the left burner.'
    out: 'Grab the red spatula on the front-left burner and place it on the left burner.'
    logit: 1.321  (base 2.642, delta -1.321)
  [put spatula on plate sink]
    in : 'Direct your manipulators to grasp the cooking implement used for scraping and sliding under food, subsequently depositing it upon the flat eating vessel that currently resides in the kitchen washbasin.'
    out: 'Pick up the toy spatula and place it on the pink plate in the sink.'
    logit: 6.365  (base 1.661, delta +4.705)
  [close brown1fbox flap]
    in : 'close the brown box lid'
    out: 'Shut the cardboard flap.'
    logit: 7.991  (base 6.358, delta +1.634)
  [put clothes in laundry machine]
    in : 'place the clothes into the open laundry machine'
    out: 'put the fabric inside the unclosed washing machine'
    logit: 6.685  (base 7.775, delta -1.090)
  [put lid on pot or pan]
    in : 'put the silver lid on the silver pot'
    out: 'place the metallic cover on the metallic vessel'
    logit: 5.703  (base 6.392, delta -0.690)
  [wipe pot with sponge]
    in : 'wipe pot with sponge'
    out: 'clean the silver metal cooking pot with the yellow sponge'
    logit: 7.068  (base 7.766, delta -0.699)
  [close small4fbox flaps]
    in : 'Please close the top of the small cardboard carton.'
    out: 'Shut the lid of the little brown box.'
    logit: 5.018  (base 3.860, delta +1.159)
  [put cup on plate]
    in : 'move the orange cup on the green plate'
    out: 'move the orange cup onto the yellow plate'
    logit: 5.732  (base 5.775, delta -0.042)
  [unfold the cloth from top to bottom]
    in : 'flatten the blue cloth from top down'
    out: 'smooth the light-blue microfiber cloth starting at the top and moving to the bottom'
    logit: 4.524  (base 5.908, delta -1.384)
  [pick up bowl and put in small4fbox]
    in : 'Provided that the small box is entirely unobstructed, the bowl, which you must first securely grasp and lift, should ultimately be relocated directly into it.'
    out: 'Pick up the light blue bowl and place it in the small box.'
    logit: 7.129  (base 3.986, delta +3.143)
  [put corn in pan which is on stove]
    in : 'place the orange carrot inside the empty pot'
    out: 'put the bright orange plastic carrot into the unoccupied silver container'
    logit: 5.387  (base 5.184, delta +0.203)