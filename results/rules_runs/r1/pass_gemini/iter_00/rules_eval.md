# Previous rulebook: measured performance

Whole rulebook applied: mean proxy logit 6.579 vs 6.408 for the unrephrased instruction (+0.170; higher is better, not a probability).

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
    out: 'Lift the blue bowl.'
    logit: 6.471  (base 4.594, delta +1.878)
  [put eggplant on plate]
    in : 'put the eggplant on the green plate'
    out: 'Put the eggplant on the green plate.'
    logit: 6.612  (base 6.634, delta -0.022)
  [end effector transition from object to object]
    in : 'Transfer the robot hand from the first piece to the second one.'
    out: 'Move the robot hand from the first piece to the second one.'
    logit: 0.212  (base 1.375, delta -1.163)
  [unfold the cloth from bottom right to top left]
    in : 'unfold the yellow cloth towards the top left'
    out: 'Unfold the yellow cloth towards the top left.'
    logit: 6.831  (base 6.751, delta +0.079)
  [put spatula on plate sink]
    in : 'Regarding the handheld implement designed for lifting food, kindly transport said item to the uppermost surface of the circular eating vessel located at the sink.'
    out: 'Put the handheld implement designed for lifting food on the circular eating vessel located at the sink.'
    logit: 2.913  (base 1.181, delta +1.732)
  [put the blue figure on the top edge of the cloth]
    in : 'Place the blue figure on the upper edge of the cloth.'
    out: 'Place the blue figure on the upper edge of the cloth.'
    logit: 5.793  (base 5.758, delta +0.035)
  [close oven]
    in : 'Shut the oven door.'
    out: 'Close the oven door.'
    logit: 8.321  (base 8.351, delta -0.029)
  [put pan in sink]
    in : 'Ascertain the whereabouts of the handled cooking vessel and subsequently ensure its physical relocation to the interior cavity of the dish-washing basin.'
    out: 'Put the handled cooking vessel in the dish-washing basin.'
    logit: 4.181  (base 4.590, delta -0.408)
  [pick up pot from sink]
    in : 'Lift the pot up out of the washbasin.'
    out: 'Lift the pot out of the washbasin.'
    logit: 0.505  (base 2.001, delta -1.496)
  [put sweet potato in pot which is in sink]
    in : "Grab the sweet potato and drop it in the pot that's resting in the sink."
    out: 'Drop the sweet potato in the pot in the sink.'
    logit: 6.761  (base 5.867, delta +0.894)
  [put carrot on cutting board]
    in : 'Assuming you have correctly identified the orange, cone-shaped vegetable, kindly execute a spatial relocation protocol to position it squarely upon the cutting board.'
    out: 'Place the orange, cone-shaped vegetable on the cutting board.'
    logit: 6.540  (base 6.932, delta -0.392)
  [put sweet potato in pot]
    in : 'Transfer the sweet potato over to the pot.'
    out: 'Put the sweet potato in the pot.'
    logit: 7.126  (base 6.928, delta +0.198)
  [put pot in sink]
    in : 'The pot should be placed in the sink.'
    out: 'Place the pot in the sink.'
    logit: 7.771  (base 7.320, delta +0.452)
  [put fork from basket to tray]
    in : 'put fork from basket to tray'
    out: 'Put the fork from the basket into the tray.'
    logit: 5.946  (base 6.915, delta -0.969)
  [take spatula off plate sink]
    in : 'Please take the spatula off the plate and put it in the sink basin.'
    out: 'Move the spatula from the plate to the sink basin.'
    logit: 7.305  (base 7.283, delta +0.022)
  [put broccoli in pot]
    in : 'put the green broccoli piece inside the metal pot'
    out: 'Put the green broccoli piece inside the metal pot.'
    logit: 6.461  (base 6.255, delta +0.206)
  [put carrot on cutting board]
    in : 'set the carrot on the cutting board'
    out: 'Put the carrot on the cutting board.'
    logit: 7.789  (base 7.805, delta -0.016)
  [zip zipper bag]
    in : 'Close the zipper bag.'
    out: 'Close the zipper bag.'
    logit: 7.654  (base 7.595, delta +0.059)
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
    out: 'Spread the orange cloth leftward.'
    logit: 5.629  (base 6.167, delta -0.538)
  [put banana in pot or pan]
    in : 'place banana in the silver pot'
    out: 'Place the banana in the silver pot.'
    logit: 6.235  (base 6.121, delta +0.114)
  [put carrot on cutting board]
    in : 'Shift the carrot to the cutting board.'
    out: 'Move the carrot to the cutting board.'
    logit: 7.842  (base 7.757, delta +0.085)
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
    out: 'Fold the bottom left corner of the orange towel to the top right.'
    logit: 6.560  (base 6.663, delta -0.104)
  [close small4fbox flaps]
    in : 'Assuming the current assembly phase requires sealing, proceed to meticulously collapse the quartet of projecting leaves atop the miniature parcel.'
    out: 'Fold down the four flaps on the small box.'
    logit: 7.751  (base 4.990, delta +2.762)
  [take the red object out of the pot and put it on the left burner]
    in : 'Targeting the red object that currently resides within the confines of the pot, please initiate a retrieval sequence to extract it, culminating in its placement atop the left burner.'
    out: 'Move the red object from the pot to the left burner.'
    logit: 1.906  (base 2.256, delta -0.349)
  [put spatula on plate sink]
    in : 'Direct your manipulators to grasp the cooking implement used for scraping and sliding under food, subsequently depositing it upon the flat eating vessel that currently resides in the kitchen washbasin.'
    out: 'Pick up the spatula and put it on the plate in the sink.'
    logit: 7.259  (base 1.805, delta +5.454)
  [close brown1fbox flap]
    in : 'close the brown box lid'
    out: 'Close the brown box lid.'
    logit: 6.231  (base 6.000, delta +0.232)
  [put clothes in laundry machine]
    in : 'place the clothes into the open laundry machine'
    out: 'Place the clothes into the open laundry machine.'
    logit: 7.783  (base 7.731, delta +0.052)
  [put lid on pot or pan]
    in : 'put the silver lid on the silver pot'
    out: 'Put the silver lid on the silver pot.'
    logit: 6.257  (base 5.984, delta +0.273)
  [wipe pot with sponge]
    in : 'wipe pot with sponge'
    out: 'Wipe the pot with the sponge.'
    logit: 7.551  (base 7.764, delta -0.212)
  [close small4fbox flaps]
    in : 'Please close the top of the small cardboard carton.'
    out: 'Close the top of the small cardboard carton.'
    logit: 4.398  (base 4.437, delta -0.039)
  [put cup on plate]
    in : 'move the orange cup on the green plate'
    out: 'Put the orange cup on the yellow plate.'
    logit: 5.754  (base 6.224, delta -0.470)
  [unfold the cloth from top to bottom]
    in : 'flatten the blue cloth from top down'
    out: 'Flatten the blue cloth from top to bottom.'
    logit: 6.329  (base 5.993, delta +0.335)
  [pick up bowl and put in small4fbox]
    in : 'Provided that the small box is entirely unobstructed, the bowl, which you must first securely grasp and lift, should ultimately be relocated directly into it.'
    out: 'Put the bowl into the small box.'
    logit: 7.200  (base 3.671, delta +3.529)
  [put corn in pan which is on stove]
    in : 'place the orange carrot inside the empty pot'
    out: 'Put the orange carrot in the empty pot.'
    logit: 4.876  (base 5.868, delta -0.993)