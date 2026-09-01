# Previous rulebook: measured performance

Whole rulebook applied: mean proxy logit 6.990 vs 6.408 for the unrephrased instruction (+0.581; higher is better, not a probability).

## Sample rewrites from the whole rulebook
(logit: higher is better; delta = rewrite - base, positive means
the rewrite beat leaving the instruction alone)

  [unfold the cloth from bottom to top]
    in : 'unfold orange cloth from bottom to top'
    out: 'unfold the cloth from bottom to top'
    logit: 7.602  (base 6.686, delta +0.916)
  [open oven]
    in : 'Swing the door to the oven open.'
    out: 'Open the door to the oven.'
    logit: 8.496  (base 8.496, delta -0.001)
  [lift bowl]
    in : 'Provided that you have secured a sufficient grip on the hemispherical dish intended for liquids, proceed to increase its altitude relative to the current resting plane.'
    out: 'Lift the bowl.'
    logit: 7.024  (base 4.897, delta +2.127)
  [put eggplant on plate]
    in : 'put the eggplant on the green plate'
    out: 'pick up the eggplant and put it on the green plate'
    logit: 6.429  (base 6.774, delta -0.346)
  [end effector transition from object to object]
    in : 'Transfer the robot hand from the first piece to the second one.'
    out: 'End effector transition from the first piece to the second one.'
    logit: 8.099  (base 1.199, delta +6.899)
  [unfold the cloth from bottom right to top left]
    in : 'unfold the yellow cloth towards the top left'
    out: 'unfold the cloth from bottom right to top left'
    logit: 7.833  (base 6.831, delta +1.002)
  [put spatula on plate sink]
    in : 'Regarding the handheld implement designed for lifting food, kindly transport said item to the uppermost surface of the circular eating vessel located at the sink.'
    out: 'Pick up the spatula and put it on the plate at the sink.'
    logit: 7.184  (base 1.665, delta +5.519)
  [put the blue figure on the top edge of the cloth]
    in : 'Place the blue figure on the upper edge of the cloth.'
    out: 'Place the blue figure on the upper edge of the cloth.'
    logit: 5.793  (base 5.793, delta +0.000)
  [close oven]
    in : 'Shut the oven door.'
    out: 'Shut the oven door.'
    logit: 8.311  (base 8.311, delta +0.000)
  [put pan in sink]
    in : 'Ascertain the whereabouts of the handled cooking vessel and subsequently ensure its physical relocation to the interior cavity of the dish-washing basin.'
    out: 'Pick up the pot and put it in the sink.'
    logit: 6.330  (base 4.620, delta +1.710)
  [pick up pot from sink]
    in : 'Lift the pot up out of the washbasin.'
    out: 'Lift the pot up out of the washbasin.'
    logit: 1.774  (base 1.774, delta +0.000)
  [put sweet potato in pot which is in sink]
    in : "Grab the sweet potato and drop it in the pot that's resting in the sink."
    out: 'Grab the sweet potato and drop it in the pot in the sink.'
    logit: 6.512  (base 6.736, delta -0.224)
  [put carrot on cutting board]
    in : 'Assuming you have correctly identified the orange, cone-shaped vegetable, kindly execute a spatial relocation protocol to position it squarely upon the cutting board.'
    out: 'put the carrot on the cutting board'
    logit: 7.849  (base 6.694, delta +1.155)
  [put sweet potato in pot]
    in : 'Transfer the sweet potato over to the pot.'
    out: 'Pick up the sweet potato and put it in the pot.'
    logit: 6.727  (base 6.958, delta -0.231)
  [put pot in sink]
    in : 'The pot should be placed in the sink.'
    out: 'Pick up the pot and put it in the sink.'
    logit: 7.679  (base 7.402, delta +0.278)
  [put fork from basket to tray]
    in : 'put fork from basket to tray'
    out: 'pick up the fork from the basket and put it in the tray'
    logit: 6.072  (base 6.320, delta -0.248)
  [take spatula off plate sink]
    in : 'Please take the spatula off the plate and put it in the sink basin.'
    out: 'Take the spatula off the plate and put it in the sink basin.'
    logit: 7.584  (base 7.581, delta +0.003)
  [put broccoli in pot]
    in : 'put the green broccoli piece inside the metal pot'
    out: 'pick up the broccoli and put it in the pot'
    logit: 6.693  (base 6.613, delta +0.080)
  [put carrot on cutting board]
    in : 'set the carrot on the cutting board'
    out: 'pick up the carrot and put it on the cutting board'
    logit: 7.838  (base 7.836, delta +0.002)
  [zip zipper bag]
    in : 'Close the zipper bag.'
    out: 'Close the bag.'
    logit: 6.605  (base 7.654, delta -1.048)
  [open fridge]
    in : 'Get the refrigerator open for me.'
    out: 'Open the refrigerator.'
    logit: 7.644  (base 7.552, delta +0.092)
  [put corn in pan which is on stove]
    in : 'In the pan on the stove, please put the corn.'
    out: 'Take the corn and put it in the pan on the stove.'
    logit: 6.292  (base 6.215, delta +0.077)
  [unfold the cloth from right to left]
    in : 'spread the orange cloth leftward'
    out: 'spread the cloth leftward'
    logit: 6.136  (base 6.046, delta +0.090)
  [put banana in pot or pan]
    in : 'place banana in the silver pot'
    out: 'pick up the banana and put it in the pot'
    logit: 6.557  (base 6.297, delta +0.260)
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
    out: 'fold the towel from bottom left to top right'
    logit: 7.725  (base 6.568, delta +1.157)
  [close small4fbox flaps]
    in : 'Assuming the current assembly phase requires sealing, proceed to meticulously collapse the quartet of projecting leaves atop the miniature parcel.'
    out: 'Fold the four flaps atop the small box.'
    logit: 7.766  (base 4.613, delta +3.153)
  [take the red object out of the pot and put it on the left burner]
    in : 'Targeting the red object that currently resides within the confines of the pot, please initiate a retrieval sequence to extract it, culminating in its placement atop the left burner.'
    out: 'take the red object out of the pot and put it on the left burner'
    logit: 7.806  (base 2.642, delta +5.163)
  [put spatula on plate sink]
    in : 'Direct your manipulators to grasp the cooking implement used for scraping and sliding under food, subsequently depositing it upon the flat eating vessel that currently resides in the kitchen washbasin.'
    out: 'Pick up the spatula and put it on the plate in the sink.'
    logit: 7.259  (base 1.661, delta +5.598)
  [close brown1fbox flap]
    in : 'close the brown box lid'
    out: 'close the box lid'
    logit: 5.809  (base 6.358, delta -0.549)
  [put clothes in laundry machine]
    in : 'place the clothes into the open laundry machine'
    out: 'pick up the clothes and put them in the laundry machine'
    logit: 7.739  (base 7.775, delta -0.036)
  [put lid on pot or pan]
    in : 'put the silver lid on the silver pot'
    out: 'pick up the lid and put it on the pot'
    logit: 5.981  (base 6.392, delta -0.411)
  [wipe pot with sponge]
    in : 'wipe pot with sponge'
    out: 'wipe the pot with the sponge'
    logit: 7.677  (base 7.766, delta -0.089)
  [close small4fbox flaps]
    in : 'Please close the top of the small cardboard carton.'
    out: 'Close the top of the carton.'
    logit: 5.095  (base 3.860, delta +1.235)
  [put cup on plate]
    in : 'move the orange cup on the green plate'
    out: 'pick up the cup and put it on the plate'
    logit: 7.088  (base 5.775, delta +1.313)
  [unfold the cloth from top to bottom]
    in : 'flatten the blue cloth from top down'
    out: 'flatten the cloth from top down'
    logit: 5.811  (base 5.908, delta -0.096)
  [pick up bowl and put in small4fbox]
    in : 'Provided that the small box is entirely unobstructed, the bowl, which you must first securely grasp and lift, should ultimately be relocated directly into it.'
    out: 'Pick up the bowl and put it in the box.'
    logit: 7.239  (base 3.986, delta +3.253)
  [put corn in pan which is on stove]
    in : 'place the orange carrot inside the empty pot'
    out: 'pick up the carrot and put it in the pot'
    logit: 5.433  (base 5.184, delta +0.249)