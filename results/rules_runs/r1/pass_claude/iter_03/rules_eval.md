# Previous rulebook: measured performance

Whole rulebook applied: mean proxy logit 6.980 vs 6.408 for the unrephrased instruction (+0.571; higher is better, not a probability).

## Sample rewrites from the whole rulebook
(logit: higher is better; delta = rewrite - base, positive means
the rewrite beat leaving the instruction alone)

  [unfold the cloth from bottom to top]
    in : 'unfold orange cloth from bottom to top'
    out: 'unfold orange cloth from bottom to top'
    logit: 6.686  (base 6.686, delta +0.000)
  [open oven]
    in : 'Swing the door to the oven open.'
    out: 'Swing the door to the oven open.'
    logit: 8.496  (base 8.496, delta +0.000)
  [lift bowl]
    in : 'Provided that you have secured a sufficient grip on the hemispherical dish intended for liquids, proceed to increase its altitude relative to the current resting plane.'
    out: 'lift the bowl above the counter'
    logit: 6.842  (base 4.897, delta +1.944)
  [put eggplant on plate]
    in : 'put the eggplant on the green plate'
    out: 'put the eggplant on the green plate'
    logit: 6.761  (base 6.774, delta -0.013)
  [end effector transition from object to object]
    in : 'Transfer the robot hand from the first piece to the second one.'
    out: 'end effector transition from the first piece to the second one'
    logit: 8.275  (base 1.199, delta +7.075)
  [unfold the cloth from bottom right to top left]
    in : 'unfold the yellow cloth towards the top left'
    out: 'unfold the yellow cloth towards the top left'
    logit: 6.831  (base 6.831, delta +0.000)
  [put spatula on plate sink]
    in : 'Regarding the handheld implement designed for lifting food, kindly transport said item to the uppermost surface of the circular eating vessel located at the sink.'
    out: 'put the spatula on top of the plate at the sink'
    logit: 7.525  (base 1.665, delta +5.860)
  [put the blue figure on the top edge of the cloth]
    in : 'Place the blue figure on the upper edge of the cloth.'
    out: 'Place the blue figure on the top edge of the cloth.'
    logit: 7.906  (base 5.793, delta +2.113)
  [close oven]
    in : 'Shut the oven door.'
    out: 'Shut the oven door.'
    logit: 8.311  (base 8.311, delta +0.000)
  [put pan in sink]
    in : 'Ascertain the whereabouts of the handled cooking vessel and subsequently ensure its physical relocation to the interior cavity of the dish-washing basin.'
    out: 'Put the pot in the sink.'
    logit: 6.032  (base 4.620, delta +1.412)
  [pick up pot from sink]
    in : 'Lift the pot up out of the washbasin.'
    out: 'Take the pot out of the sink.'
    logit: 4.958  (base 1.774, delta +3.184)
  [put sweet potato in pot which is in sink]
    in : "Grab the sweet potato and drop it in the pot that's resting in the sink."
    out: "Grab the sweet potato and drop it in the pot that's resting in the sink."
    logit: 6.736  (base 6.736, delta +0.000)
  [put carrot on cutting board]
    in : 'Assuming you have correctly identified the orange, cone-shaped vegetable, kindly execute a spatial relocation protocol to position it squarely upon the cutting board.'
    out: 'Put the orange vegetable on the cutting board.'
    logit: 7.135  (base 6.694, delta +0.440)
  [put sweet potato in pot]
    in : 'Transfer the sweet potato over to the pot.'
    out: 'Move the sweet potato over to the pot.'
    logit: 6.771  (base 6.958, delta -0.187)
  [put pot in sink]
    in : 'The pot should be placed in the sink.'
    out: 'put the pot in the sink'
    logit: 7.702  (base 7.402, delta +0.300)
  [put fork from basket to tray]
    in : 'put fork from basket to tray'
    out: 'put fork from basket to tray'
    logit: 6.320  (base 6.320, delta +0.000)
  [take spatula off plate sink]
    in : 'Please take the spatula off the plate and put it in the sink basin.'
    out: 'Take the spatula off the plate and put it in the sink basin.'
    logit: 7.584  (base 7.581, delta +0.003)
  [put broccoli in pot]
    in : 'put the green broccoli piece inside the metal pot'
    out: 'put the broccoli piece inside the pot'
    logit: 6.993  (base 6.613, delta +0.380)
  [put carrot on cutting board]
    in : 'set the carrot on the cutting board'
    out: 'set the carrot on the cutting board'
    logit: 7.836  (base 7.836, delta +0.000)
  [zip zipper bag]
    in : 'Close the zipper bag.'
    out: 'Close the zipper bag.'
    logit: 7.654  (base 7.654, delta +0.000)
  [open fridge]
    in : 'Get the refrigerator open for me.'
    out: 'Open the fridge.'
    logit: 7.663  (base 7.552, delta +0.111)
  [put corn in pan which is on stove]
    in : 'In the pan on the stove, please put the corn.'
    out: 'Put the corn in the pan on the stove.'
    logit: 6.549  (base 6.215, delta +0.334)
  [unfold the cloth from right to left]
    in : 'spread the orange cloth leftward'
    out: 'spread the cloth leftward'
    logit: 6.136  (base 6.046, delta +0.090)
  [put banana in pot or pan]
    in : 'place banana in the silver pot'
    out: 'put the banana in the pot'
    logit: 7.710  (base 6.297, delta +1.413)
  [put carrot on cutting board]
    in : 'Shift the carrot to the cutting board.'
    out: 'Shift the carrot to the cutting board.'
    logit: 7.854  (base 7.854, delta +0.000)
  [put pan in sink]
    in : 'put pan into the sink'
    out: 'put pan into the sink'
    logit: 7.467  (base 7.467, delta +0.000)
  [close the drawer]
    in : 'Shut the drawer.'
    out: 'Shut the drawer.'
    logit: 8.650  (base 8.650, delta +0.000)
  [fold the cloth from bottom left to top right]
    in : 'grab the bottom left corner of the orange towel and fold it top right'
    out: 'grab the bottom left corner of the orange towel and fold it top right'
    logit: 6.568  (base 6.568, delta +0.000)
  [close small4fbox flaps]
    in : 'Assuming the current assembly phase requires sealing, proceed to meticulously collapse the quartet of projecting leaves atop the miniature parcel.'
    out: 'fold the four flaps atop the box'
    logit: 7.801  (base 4.613, delta +3.188)
  [take the red object out of the pot and put it on the left burner]
    in : 'Targeting the red object that currently resides within the confines of the pot, please initiate a retrieval sequence to extract it, culminating in its placement atop the left burner.'
    out: 'Take the red object out of the pot and place it atop the left burner.'
    logit: 5.228  (base 2.642, delta +2.586)
  [put spatula on plate sink]
    in : 'Direct your manipulators to grasp the cooking implement used for scraping and sliding under food, subsequently depositing it upon the flat eating vessel that currently resides in the kitchen washbasin.'
    out: 'Pick up the spatula and put it upon the plate that is in the sink.'
    logit: 7.168  (base 1.661, delta +5.507)
  [close brown1fbox flap]
    in : 'close the brown box lid'
    out: 'close the box lid'
    logit: 5.809  (base 6.358, delta -0.549)
  [put clothes in laundry machine]
    in : 'place the clothes into the open laundry machine'
    out: 'put the clothes into the washing machine'
    logit: 7.874  (base 7.775, delta +0.099)
  [put lid on pot or pan]
    in : 'put the silver lid on the silver pot'
    out: 'put the lid on the pot'
    logit: 6.175  (base 6.392, delta -0.217)
  [wipe pot with sponge]
    in : 'wipe pot with sponge'
    out: 'wipe pot with sponge'
    logit: 7.766  (base 7.766, delta +0.000)
  [close small4fbox flaps]
    in : 'Please close the top of the small cardboard carton.'
    out: 'Close the top of the small cardboard carton.'
    logit: 4.398  (base 3.860, delta +0.539)
  [put cup on plate]
    in : 'move the orange cup on the green plate'
    out: 'move the orange cup on the green plate'
    logit: 5.775  (base 5.775, delta +0.000)
  [unfold the cloth from top to bottom]
    in : 'flatten the blue cloth from top down'
    out: 'flatten the blue cloth from top down'
    logit: 5.908  (base 5.908, delta +0.000)
  [pick up bowl and put in small4fbox]
    in : 'Provided that the small box is entirely unobstructed, the bowl, which you must first securely grasp and lift, should ultimately be relocated directly into it.'
    out: 'Grasp and lift the bowl and put it into the box.'
    logit: 7.357  (base 3.986, delta +3.370)
  [put corn in pan which is on stove]
    in : 'place the orange carrot inside the empty pot'
    out: 'put the carrot in the pot'
    logit: 5.380  (base 5.184, delta +0.196)