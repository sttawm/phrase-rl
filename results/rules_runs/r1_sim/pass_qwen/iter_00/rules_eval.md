# Previous rulebook: measured performance

Whole rulebook applied: mean proxy logit 0.306 vs 0.229 for the unrephrased instruction (+0.076; higher is better, not a probability).

## Sample rewrites from the whole rulebook
(logit: higher is better; delta = rewrite - base, positive means
the rewrite beat leaving the instruction alone)

  [widowx_coke_can_on_plate_clean]
    in : 'put the coke down on the plate'
    out: 'put the coke on the plate'
    logit: 50.000  (base -0.237, delta +50.237)
  [widowx_carrot_on_wheel_clean]
    in : 'Transfer the carrot over to the wheel.'
    out: 'Place the carrot on the wheel.'
    logit: 33.333  (base -4.170, delta +37.503)
  [widowx_carrot_on_plate]
    in : 'put the carrot on top of the plate'
    out: 'put the carrot on the plate'
    logit: 16.667  (base 2.518, delta +14.149)
  [widowx_carrot_on_keyboard_clean]
    in : 'set the carrot on the keyboard'
    out: 'put the carrot on the keyboard'
    logit: 0.000  (base -1.139, delta +1.139)
  [widowx_carrot_on_wheel_clean]
    in : 'set the carrot on the wheel'
    out: 'put the carrot on the wheel'
    logit: 0.000  (base -2.849, delta +2.849)
  [widowx_put_eggplant_in_basket]
    in : 'Slide the eggplant into the yellow basket.'
    out: 'Slide the purple eggplant into the yellow basket.'
    logit: 100.000  (base 3.237, delta +96.763)
  [widowx_coke_can_on_plate_clean]
    in : 'place coke can on plate'
    out: 'place coke can on plate'
    logit: 33.333  (base -0.037, delta +33.370)
  [widowx_carrot_on_plate]
    in : 'Please shift the carrot onto the plate.'
    out: 'Shift the carrot onto the plate.'
    logit: 33.333  (base 2.198, delta +31.135)
  [widowx_carrot_on_keyboard_clean]
    in : 'Pick up the carrot and set it down on the keyboard.'
    out: 'Pick up the carrot and set it down on the keyboard.'
    logit: 0.000  (base -1.232, delta +1.232)
  [widowx_carrot_on_keyboard_clean]
    in : 'place the carrot across the black keyboard'
    out: 'place the carrot across the keyboard'
    logit: 16.667  (base -0.020, delta +16.687)
  [widowx_coke_can_on_ramekin_clean]
    in : 'Pick up the red cola can and place it upright inside the white bowl.'
    out: 'Pick up the red cola can and place it upright inside the white bowl.'
    logit: 50.000  (base 0.541, delta +49.459)
  [widowx_coke_can_on_ramekin_clean]
    in : 'place the cola can standing up inside the white bowl'
    out: 'place the red soda can upright in the white bowl'
    logit: 50.000  (base 0.092, delta +49.908)
  [widowx_coke_can_on_plate_clean]
    in : 'Grab that red aluminum can and set it on the plate.'
    out: 'Grab the red soda can and put it on the plate.'
    logit: 0.000  (base 0.076, delta -0.076)
  [widowx_spoon_on_towel]
    in : 'Could you grab that spoon and set it on the cloth for me?'
    out: 'Pick up the green-handled spoon and put it on the blue cloth.'
    logit: 50.000  (base 1.856, delta +48.144)
  [widowx_stack_cube]
    in : 'Stack the green on the yellow.'
    out: 'Stack the green block on the yellow block.'
    logit: 66.667  (base -2.108, delta +68.774)
  [widowx_stack_cube]
    in : 'green block on yellow block'
    out: 'put green block on yellow block'
    logit: 33.333  (base -2.281, delta +35.615)
  [widowx_stack_cube]
    in : 'Can you place the green block onto the yellow block for me?'
    out: 'Stack the green block on the yellow block.'
    logit: 66.667  (base -2.526, delta +69.193)
  [widowx_coke_can_on_plate_clean]
    in : 'put the coke can on the dish'
    out: 'put the coke can on the dish'
    logit: 0.000  (base -1.027, delta +1.027)
  [widowx_stack_cube]
    in : 'put the green cube on the yellow one'
    out: 'put the green cube on the yellow one'
    logit: 33.333  (base -2.363, delta +35.696)
  [widowx_coke_can_on_ramekin_clean]
    in : 'place the cola can inside the white bowl'
    out: 'place the red can in the white bowl'
    logit: 50.000  (base 0.578, delta +49.422)
  [widowx_coke_can_on_ramekin_clean]
    in : 'place the red can upright inside the white bowl'
    out: 'put the red can upright in the white bowl'
    logit: 33.333  (base 0.306, delta +33.028)
  [widowx_spoon_on_towel]
    in : 'put the spoon on the blue cloth'
    out: 'put the spoon on the blue cloth'
    logit: 16.667  (base 2.381, delta +14.286)
  [widowx_carrot_on_wheel_clean]
    in : 'place the carrot in the black wheel'
    out: 'place the carrot in the black wheel'
    logit: 16.667  (base -1.809, delta +18.476)
  [widowx_stack_cube]
    in : 'Put the green one on top of the yellow one.'
    out: 'Put the green block on top of the yellow block.'
    logit: 16.667  (base -1.981, delta +18.648)
  [widowx_carrot_on_keyboard_clean]
    in : 'Lay the carrot across the keyboard.'
    out: 'Lay the carrot across the keyboard.'
    logit: 0.000  (base -0.895, delta +0.895)
  [widowx_carrot_on_wheel_clean]
    in : 'Move the orange vegetable to the wheel.'
    out: 'Move the orange vegetable to the wheel.'
    logit: 0.000  (base -3.071, delta +3.071)
  [widowx_spoon_on_towel]
    in : 'It would be highly appreciated if you could relocate the concave piece of silverware so that it resides atop the woven textile.'
    out: 'Pick up the green-handled spoon and put it on the blue cloth.'
    logit: 50.000  (base 1.131, delta +48.868)
  [widowx_carrot_on_keyboard_clean]
    in : 'Assuming you have grasped the orange root vegetable, please transport it such that it rests atop the device used for typing.'
    out: 'Place the carrot on the keyboard.'
    logit: 0.000  (base -2.190, delta +2.190)
  [widowx_coke_can_on_ramekin_clean]
    in : 'Should you be capable of manipulating the cylindrical cola container, proceed to hoist it and guide it downward until it is securely housed in the ramekin.'
    out: 'Pick up the red can and put it in the white ramekin.'
    logit: 16.667  (base -1.342, delta +18.009)
  [widowx_carrot_on_plate]
    in : 'You are hereby instructed to commence the relocation of the carrot, terminating said motion only when it makes contact with the plate.'
    out: 'Pick up the carrot and put it on the plate.'
    logit: 16.667  (base 0.903, delta +15.763)
  [widowx_carrot_on_plate]
    in : 'Proceed to grasp the carrot and subsequently establish its position atop the dining plate.'
    out: 'Pick up the carrot and put it on the plate.'
    logit: 16.667  (base 1.307, delta +15.360)
  [widowx_carrot_on_plate]
    in : 'The carrot, which is presently situated in its initial starting location, must be lifted and maneuvered until it occupies the plate.'
    out: 'Pick up the carrot and place it on the plate.'
    logit: 33.333  (base 0.607, delta +32.726)
  [widowx_carrot_on_plate]
    in : 'Given the current arrangement of objects, ensure that the carrot is swiftly conveyed to the surface of the plate.'
    out: 'Convey the carrot to the plate.'
    logit: 50.000  (base 1.537, delta +48.464)
  [widowx_carrot_on_wheel_clean]
    in : 'The elongated orange taproot must be elevated from its current location and meticulously positioned such that it is fully supported by the wheel.'
    out: 'Place the carrot on the wheel.'
    logit: 33.333  (base -6.400, delta +39.733)
  [widowx_stack_cube]
    in : 'Avoiding all other objects, ensure that the highest point of the yellow block becomes the new foundation for the green block.'
    out: 'Stack the green block on the yellow block.'
    logit: 66.667  (base -4.703, delta +71.370)
  [widowx_coke_can_on_plate_clean]
    in : 'Assuming the plate is readily accessible, proceed to transport the red tin of cola directly onto its surface.'
    out: 'Pick up the red tin of cola and put it on the plate.'
    logit: 0.000  (base -3.406, delta +3.405)
  [widowx_stack_cube]
    in : 'Assuming the yellow block is currently clear, elevate the green block until it can be deposited directly upon its upper surface.'
    out: 'Pick up the green block and put it on the yellow block.'
    logit: 50.000  (base -4.100, delta +54.100)
  [widowx_stack_cube]
    in : 'Without disturbing the surrounding items, it is required that the block of green coloration be made to rest atop its yellow counterpart.'
    out: 'Pick up the green block and place it on the yellow block.'
    logit: 33.333  (base -3.575, delta +36.909)
  [widowx_put_eggplant_in_basket]
    in : 'Deposit the bulbous purple vegetable directly into the confines of the yellow woven basket.'
    out: 'Put the purple vegetable in the yellow basket.'
    logit: 100.000  (base 1.790, delta +98.210)
  [widowx_spoon_on_towel]
    in : 'It is hereby requested that the spoon be promptly elevated from its current station and subsequently lowered onto the surface of the towel.'
    out: 'Pick up the spoon and put it on the towel.'
    logit: 16.667  (base 0.509, delta +16.157)