# Previous rulebook: measured performance

Whole rulebook applied: mean proxy logit 0.333 vs 0.229 for the unrephrased instruction (+0.104; higher is better, not a probability).

## Sample rewrites from the whole rulebook
(logit: higher is better; delta = rewrite - base, positive means
the rewrite beat leaving the instruction alone)

  [widowx_coke_can_on_plate_clean]
    in : 'put the coke down on the plate'
    out: 'put the coke on the plate'
    logit: 50.000  (base 83.333, delta -33.333)
  [widowx_carrot_on_wheel_clean]
    in : 'Transfer the carrot over to the wheel.'
    out: 'Put the orange carrot on the wheel.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_carrot_on_plate]
    in : 'put the carrot on top of the plate'
    out: 'put the carrot on top of the plate'
    logit: 33.333  (base 33.333, delta +0.000)
  [widowx_carrot_on_keyboard_clean]
    in : 'set the carrot on the keyboard'
    out: 'Place the carrot on the keyboard.'
    logit: 33.333  (base 0.000, delta +33.333)
  [widowx_carrot_on_wheel_clean]
    in : 'set the carrot on the wheel'
    out: 'Place the carrot on the wheel.'
    logit: 33.333  (base 16.667, delta +16.667)
  [widowx_put_eggplant_in_basket]
    in : 'Slide the eggplant into the yellow basket.'
    out: 'Slide the purple eggplant into the yellow basket.'
    logit: 100.000  (base 83.333, delta +16.667)
  [widowx_coke_can_on_plate_clean]
    in : 'place coke can on plate'
    out: 'place coke can on plate'
    logit: 33.333  (base 50.000, delta -16.667)
  [widowx_carrot_on_plate]
    in : 'Please shift the carrot onto the plate.'
    out: 'Put the carrot on the plate.'
    logit: 33.333  (base 16.667, delta +16.667)
  [widowx_carrot_on_keyboard_clean]
    in : 'Pick up the carrot and set it down on the keyboard.'
    out: 'Put the orange carrot on the black keyboard.'
    logit: 16.667  (base 0.000, delta +16.667)
  [widowx_carrot_on_keyboard_clean]
    in : 'place the carrot across the black keyboard'
    out: 'Place the carrot across the black keyboard.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_coke_can_on_ramekin_clean]
    in : 'Pick up the red cola can and place it upright inside the white bowl.'
    out: 'Pick up the red cola can and place it upright inside the white bowl.'
    logit: 50.000  (base 50.000, delta +0.000)
  [widowx_coke_can_on_ramekin_clean]
    in : 'place the cola can standing up inside the white bowl'
    out: 'Place the red cola can standing up inside the white bowl.'
    logit: 33.333  (base 50.000, delta -16.667)
  [widowx_coke_can_on_plate_clean]
    in : 'Grab that red aluminum can and set it on the plate.'
    out: 'Grab the red aluminum can and put it on the light green plate.'
    logit: 16.667  (base 66.667, delta -50.000)
  [widowx_spoon_on_towel]
    in : 'Could you grab that spoon and set it on the cloth for me?'
    out: 'Pick up the green-handled spoon and place it on the blue cloth.'
    logit: 50.000  (base 50.000, delta +0.000)
  [widowx_stack_cube]
    in : 'Stack the green on the yellow.'
    out: 'Stack the green cube on the yellow cube.'
    logit: 50.000  (base 50.000, delta +0.000)
  [widowx_stack_cube]
    in : 'green block on yellow block'
    out: 'Put the green block on the yellow block.'
    logit: 33.333  (base 16.667, delta +16.667)
  [widowx_stack_cube]
    in : 'Can you place the green block onto the yellow block for me?'
    out: 'Place the green cube on top of the yellow cube.'
    logit: 66.667  (base 33.333, delta +33.333)
  [widowx_coke_can_on_plate_clean]
    in : 'put the coke can on the dish'
    out: 'put the red coke can on the dish'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_stack_cube]
    in : 'put the green cube on the yellow one'
    out: 'put the green cube on the yellow one'
    logit: 33.333  (base 33.333, delta +0.000)
  [widowx_coke_can_on_ramekin_clean]
    in : 'place the cola can inside the white bowl'
    out: 'Place the red can in the white bowl.'
    logit: 66.667  (base 0.000, delta +66.667)
  [widowx_coke_can_on_ramekin_clean]
    in : 'place the red can upright inside the white bowl'
    out: 'Place the red can in the white bowl.'
    logit: 66.667  (base 50.000, delta +16.667)
  [widowx_spoon_on_towel]
    in : 'put the spoon on the blue cloth'
    out: 'put the spoon on the blue cloth'
    logit: 33.333  (base 50.000, delta -16.667)
  [widowx_carrot_on_wheel_clean]
    in : 'place the carrot in the black wheel'
    out: 'Place the orange carrot in the black wheel.'
    logit: 33.333  (base 33.333, delta +0.000)
  [widowx_stack_cube]
    in : 'Put the green one on top of the yellow one.'
    out: 'Put the green block on top of the yellow block.'
    logit: 16.667  (base 33.333, delta -16.667)
  [widowx_carrot_on_keyboard_clean]
    in : 'Lay the carrot across the keyboard.'
    out: 'Place the carrot on top of the keyboard.'
    logit: 16.667  (base 0.000, delta +16.667)
  [widowx_carrot_on_wheel_clean]
    in : 'Move the orange vegetable to the wheel.'
    out: 'Place the orange vegetable on the wheel.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_spoon_on_towel]
    in : 'It would be highly appreciated if you could relocate the concave piece of silverware so that it resides atop the woven textile.'
    out: 'Place the green-handled spoon on top of the blue cloth.'
    logit: 16.667  (base 50.000, delta -33.333)
  [widowx_carrot_on_keyboard_clean]
    in : 'Assuming you have grasped the orange root vegetable, please transport it such that it rests atop the device used for typing.'
    out: 'Put the orange carrot on top of the black keyboard.'
    logit: 16.667  (base 0.000, delta +16.667)
  [widowx_coke_can_on_ramekin_clean]
    in : 'Should you be capable of manipulating the cylindrical cola container, proceed to hoist it and guide it downward until it is securely housed in the ramekin.'
    out: 'Pick up the red can and put it in the white ramekin.'
    logit: 16.667  (base 0.000, delta +16.667)
  [widowx_carrot_on_plate]
    in : 'You are hereby instructed to commence the relocation of the carrot, terminating said motion only when it makes contact with the plate.'
    out: 'Put the orange carrot on the green plate.'
    logit: 33.333  (base 16.667, delta +16.667)
  [widowx_carrot_on_plate]
    in : 'Proceed to grasp the carrot and subsequently establish its position atop the dining plate.'
    out: 'Pick up the carrot and place it on top of the dining plate.'
    logit: 33.333  (base 33.333, delta +0.000)
  [widowx_carrot_on_plate]
    in : 'The carrot, which is presently situated in its initial starting location, must be lifted and maneuvered until it occupies the plate.'
    out: 'Pick up the orange carrot and place it on the yellow plate.'
    logit: 16.667  (base 16.667, delta +0.000)
  [widowx_carrot_on_plate]
    in : 'Given the current arrangement of objects, ensure that the carrot is swiftly conveyed to the surface of the plate.'
    out: 'Put the orange carrot on the light green plate.'
    logit: 33.333  (base 16.667, delta +16.667)
  [widowx_carrot_on_wheel_clean]
    in : 'The elongated orange taproot must be elevated from its current location and meticulously positioned such that it is fully supported by the wheel.'
    out: 'Place the elongated orange taproot on the wheel.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_stack_cube]
    in : 'Avoiding all other objects, ensure that the highest point of the yellow block becomes the new foundation for the green block.'
    out: 'Stack the green block on top of the yellow block.'
    logit: 16.667  (base 0.000, delta +16.667)
  [widowx_coke_can_on_plate_clean]
    in : 'Assuming the plate is readily accessible, proceed to transport the red tin of cola directly onto its surface.'
    out: 'Put the red tin of cola on the green plate.'
    logit: 33.333  (base 0.000, delta +33.333)
  [widowx_stack_cube]
    in : 'Assuming the yellow block is currently clear, elevate the green block until it can be deposited directly upon its upper surface.'
    out: 'Pick up the green block and place it on top of the yellow block.'
    logit: 33.333  (base 0.000, delta +33.333)
  [widowx_stack_cube]
    in : 'Without disturbing the surrounding items, it is required that the block of green coloration be made to rest atop its yellow counterpart.'
    out: 'Stack the green cube on top of the yellow cube.'
    logit: 66.667  (base 0.000, delta +66.667)
  [widowx_put_eggplant_in_basket]
    in : 'Deposit the bulbous purple vegetable directly into the confines of the yellow woven basket.'
    out: 'Put the bulbous purple vegetable in the yellow basket.'
    logit: 100.000  (base 100.000, delta +0.000)
  [widowx_spoon_on_towel]
    in : 'It is hereby requested that the spoon be promptly elevated from its current station and subsequently lowered onto the surface of the towel.'
    out: 'Pick up the green spoon and put it on the blue towel.'
    logit: 33.333  (base 0.000, delta +33.333)