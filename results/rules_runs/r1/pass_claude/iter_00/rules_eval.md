# Previous rulebook: measured performance

Whole rulebook applied: mean proxy logit 6.510 vs 6.408 for the unrephrased instruction (+0.102; higher is better, not a probability).

## Sample rewrites from the whole rulebook
(logit: higher is better; delta = rewrite - base, positive means
the rewrite beat leaving the instruction alone)

  [unfold the cloth from bottom to top]
    in : 'unfold orange cloth from bottom to top'
    out: 'Unfold the orange cloth from bottom to top.'
    logit: 6.420  (base 6.612, delta -0.192)
  [open oven]
    in : 'Swing the door to the oven open.'
    out: 'Open the oven door.'
    logit: 8.524  (base 8.436, delta +0.088)
  [lift bowl]
    in : 'Provided that you have secured a sufficient grip on the hemispherical dish intended for liquids, proceed to increase its altitude relative to the current resting plane.'
    out: 'Grab the blue bowl and lift it up off the countertop.'
    logit: 6.030  (base 4.594, delta +1.436)
  [put eggplant on plate]
    in : 'put the eggplant on the green plate'
    out: 'Put the eggplant on the green plate.'
    logit: 6.612  (base 6.634, delta -0.022)
  [end effector transition from object to object]
    in : 'Transfer the robot hand from the first piece to the second one.'
    out: 'Move the gripper from the orange cup to the silver colander.'
    logit: 2.677  (base 1.375, delta +1.302)
  [unfold the cloth from bottom right to top left]
    in : 'unfold the yellow cloth towards the top left'
    out: 'Unfold the yellow cloth to the upper left.'
    logit: 6.604  (base 6.751, delta -0.147)
  [put spatula on plate sink]
    in : 'Regarding the handheld implement designed for lifting food, kindly transport said item to the uppermost surface of the circular eating vessel located at the sink.'
    out: 'Move the spatula onto the pink plate in the sink.'
    logit: 7.059  (base 1.181, delta +5.878)
  [put the blue figure on the top edge of the cloth]
    in : 'Place the blue figure on the upper edge of the cloth.'
    out: 'Put the blue figure on the top edge of the cloth.'
    logit: 7.863  (base 5.758, delta +2.105)
  [close oven]
    in : 'Shut the oven door.'
    out: 'Close the oven door.'
    logit: 8.321  (base 8.351, delta -0.029)
  [put pan in sink]
    in : 'Ascertain the whereabouts of the handled cooking vessel and subsequently ensure its physical relocation to the interior cavity of the dish-washing basin.'
    out: 'Pick up the silver pot and put it in the sink.'
    logit: 6.319  (base 4.590, delta +1.730)
  [pick up pot from sink]
    in : 'Lift the pot up out of the washbasin.'
    out: 'Take the pot out of the washbasin.'
    logit: -0.290  (base 2.001, delta -2.291)
  [put sweet potato in pot which is in sink]
    in : "Grab the sweet potato and drop it in the pot that's resting in the sink."
    out: 'Put the sweet potato in the pot in the sink.'
    logit: 6.853  (base 5.867, delta +0.986)
  [put carrot on cutting board]
    in : 'Assuming you have correctly identified the orange, cone-shaped vegetable, kindly execute a spatial relocation protocol to position it squarely upon the cutting board.'
    out: 'Move the toy carrot onto the green cutting board.'
    logit: 7.378  (base 6.932, delta +0.447)
  [put sweet potato in pot]
    in : 'Transfer the sweet potato over to the pot.'
    out: 'Put the sweet potato in the pot.'
    logit: 7.126  (base 6.928, delta +0.198)
  [put pot in sink]
    in : 'The pot should be placed in the sink.'
    out: 'Put the pot in the sink.'
    logit: 7.634  (base 7.320, delta +0.314)
  [put fork from basket to tray]
    in : 'put fork from basket to tray'
    out: 'Move the fork from the basket to the tray.'
    logit: 6.093  (base 6.915, delta -0.822)
  [take spatula off plate sink]
    in : 'Please take the spatula off the plate and put it in the sink basin.'
    out: 'Take the spatula off the plate and put it in the sink basin.'
    logit: 7.584  (base 7.283, delta +0.300)
  [put broccoli in pot]
    in : 'put the green broccoli piece inside the metal pot'
    out: 'Put the green broccoli piece in the metal pot.'
    logit: 6.384  (base 6.255, delta +0.129)
  [put carrot on cutting board]
    in : 'set the carrot on the cutting board'
    out: 'Put the carrot on the cutting board.'
    logit: 7.789  (base 7.805, delta -0.016)
  [zip zipper bag]
    in : 'Close the zipper bag.'
    out: 'Zip the bag shut.'
    logit: 7.498  (base 7.595, delta -0.097)
  [open fridge]
    in : 'Get the refrigerator open for me.'
    out: 'Open the refrigerator.'
    logit: 7.644  (base 7.857, delta -0.214)
  [put corn in pan which is on stove]
    in : 'In the pan on the stove, please put the corn.'
    out: 'Put the corn in the pan on the stove.'
    logit: 6.549  (base 6.588, delta -0.040)
  [unfold the cloth from right to left]
    in : 'spread the orange cloth leftward'
    out: 'Spread the yellow cloth to the left.'
    logit: 5.977  (base 6.167, delta -0.190)
  [put banana in pot or pan]
    in : 'place banana in the silver pot'
    out: 'Place the banana in the silver pot.'
    logit: 6.235  (base 6.121, delta +0.114)
  [put carrot on cutting board]
    in : 'Shift the carrot to the cutting board.'
    out: 'Move the carrot onto the cutting board.'
    logit: 7.823  (base 7.757, delta +0.067)
  [put pan in sink]
    in : 'put pan into the sink'
    out: 'Put the pan into the sink.'
    logit: 7.483  (base 7.559, delta -0.076)
  [close the drawer]
    in : 'Shut the drawer.'
    out: 'Close the drawer.'
    logit: 8.585  (base 8.243, delta +0.342)
  [fold the cloth from bottom left to top right]
    in : 'grab the bottom left corner of the orange towel and fold it top right'
    out: 'Grab the bottom left corner of the orange towel and fold it to the top right.'
    logit: 6.488  (base 6.663, delta -0.176)
  [close small4fbox flaps]
    in : 'Assuming the current assembly phase requires sealing, proceed to meticulously collapse the quartet of projecting leaves atop the miniature parcel.'
    out: 'Fold down the four flaps on top of the small cardboard box.'
    logit: 7.729  (base 4.990, delta +2.739)
  [take the red object out of the pot and put it on the left burner]
    in : 'Targeting the red object that currently resides within the confines of the pot, please initiate a retrieval sequence to extract it, culminating in its placement atop the left burner.'
    out: 'Grab the red spatula and place it on the left burner.'
    logit: 1.437  (base 2.256, delta -0.819)
  [put spatula on plate sink]
    in : 'Direct your manipulators to grasp the cooking implement used for scraping and sliding under food, subsequently depositing it upon the flat eating vessel that currently resides in the kitchen washbasin.'
    out: 'Pick up the spatula and place it on the pink plate in the sink.'
    logit: 6.595  (base 1.805, delta +4.791)
  [close brown1fbox flap]
    in : 'close the brown box lid'
    out: 'Close the brown box lid.'
    logit: 6.231  (base 6.000, delta +0.232)
  [put clothes in laundry machine]
    in : 'place the clothes into the open laundry machine'
    out: 'Put the clothes into the washing machine.'
    logit: 7.887  (base 7.731, delta +0.156)
  [put lid on pot or pan]
    in : 'put the silver lid on the silver pot'
    out: 'Put the lid on the pot.'
    logit: 6.237  (base 5.984, delta +0.253)
  [wipe pot with sponge]
    in : 'wipe pot with sponge'
    out: 'Wipe the pot with the sponge.'
    logit: 7.551  (base 7.764, delta -0.212)
  [close small4fbox flaps]
    in : 'Please close the top of the small cardboard carton.'
    out: 'Close the top flap of the small cardboard box.'
    logit: 6.906  (base 4.437, delta +2.469)
  [put cup on plate]
    in : 'move the orange cup on the green plate'
    out: 'Put the orange cup on the plate.'
    logit: 7.031  (base 6.224, delta +0.807)
  [unfold the cloth from top to bottom]
    in : 'flatten the blue cloth from top down'
    out: 'Smooth out the blue cloth on the stove, starting at the top and moving to the bottom.'
    logit: 5.075  (base 5.993, delta -0.919)
  [pick up bowl and put in small4fbox]
    in : 'Provided that the small box is entirely unobstructed, the bowl, which you must first securely grasp and lift, should ultimately be relocated directly into it.'
    out: 'If the small box is empty, pick up the bowl and place it in the box.'
    logit: 6.251  (base 3.671, delta +2.581)
  [put corn in pan which is on stove]
    in : 'place the orange carrot inside the empty pot'
    out: 'Put the carrot in the pot.'
    logit: 5.431  (base 5.868, delta -0.437)