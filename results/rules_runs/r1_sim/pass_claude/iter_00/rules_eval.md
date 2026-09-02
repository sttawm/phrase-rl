# Previous rulebook: measured performance

Whole rulebook applied: mean proxy logit 0.278 vs 0.229 for the unrephrased instruction (+0.049; higher is better, not a probability).

## Sample rewrites from the whole rulebook
(logit: higher is better; delta = rewrite - base, positive means
the rewrite beat leaving the instruction alone)

  [widowx_coke_can_on_plate_clean]
    in : 'put the coke down on the plate'
    out: 'Put the coke on the plate.'
    logit: 66.667  (base 83.333, delta -16.667)
  [widowx_carrot_on_wheel_clean]
    in : 'Transfer the carrot over to the wheel.'
    out: 'Put the carrot in the center of the wheel.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_carrot_on_plate]
    in : 'put the carrot on top of the plate'
    out: 'Put the carrot on the plate.'
    logit: 33.333  (base 33.333, delta +0.000)
  [widowx_carrot_on_keyboard_clean]
    in : 'set the carrot on the keyboard'
    out: 'set the carrot on the keyboard'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_carrot_on_wheel_clean]
    in : 'set the carrot on the wheel'
    out: 'Set the carrot on the wheel.'
    logit: 16.667  (base 16.667, delta +0.000)
  [widowx_put_eggplant_in_basket]
    in : 'Slide the eggplant into the yellow basket.'
    out: 'Slide the eggplant into the yellow basket.'
    logit: 83.333  (base 83.333, delta +0.000)
  [widowx_coke_can_on_plate_clean]
    in : 'place coke can on plate'
    out: 'Put the coke can on the plate.'
    logit: 33.333  (base 50.000, delta -16.667)
  [widowx_carrot_on_plate]
    in : 'Please shift the carrot onto the plate.'
    out: 'Move the carrot onto the plate.'
    logit: 33.333  (base 16.667, delta +16.667)
  [widowx_carrot_on_keyboard_clean]
    in : 'Pick up the carrot and set it down on the keyboard.'
    out: 'Put the carrot on the keyboard.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_carrot_on_keyboard_clean]
    in : 'place the carrot across the black keyboard'
    out: 'Place the carrot across the black keyboard.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_coke_can_on_ramekin_clean]
    in : 'Pick up the red cola can and place it upright inside the white bowl.'
    out: 'Grab the red cola can and set it upright in the white bowl.'
    logit: 50.000  (base 50.000, delta +0.000)
  [widowx_coke_can_on_ramekin_clean]
    in : 'place the cola can standing up inside the white bowl'
    out: 'Place the cola can upright in the white bowl.'
    logit: 0.000  (base 50.000, delta -50.000)
  [widowx_coke_can_on_plate_clean]
    in : 'Grab that red aluminum can and set it on the plate.'
    out: 'Pick up the red can and place it on the green plate.'
    logit: 33.333  (base 66.667, delta -33.333)
  [widowx_spoon_on_towel]
    in : 'Could you grab that spoon and set it on the cloth for me?'
    out: 'Pick up the green spoon and place it on the blue cloth.'
    logit: 33.333  (base 50.000, delta -16.667)
  [widowx_stack_cube]
    in : 'Stack the green on the yellow.'
    out: 'Put the green cube on the yellow cube.'
    logit: 33.333  (base 50.000, delta -16.667)
  [widowx_stack_cube]
    in : 'green block on yellow block'
    out: 'Stack the green cube on the yellow cube.'
    logit: 66.667  (base 16.667, delta +50.000)
  [widowx_stack_cube]
    in : 'Can you place the green block onto the yellow block for me?'
    out: 'Place the green block onto the yellow block.'
    logit: 33.333  (base 33.333, delta +0.000)
  [widowx_coke_can_on_plate_clean]
    in : 'put the coke can on the dish'
    out: 'Put the coke can on the dish.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_stack_cube]
    in : 'put the green cube on the yellow one'
    out: 'Stack the green cube on the yellow cube.'
    logit: 66.667  (base 33.333, delta +33.333)
  [widowx_coke_can_on_ramekin_clean]
    in : 'place the cola can inside the white bowl'
    out: 'Put the cola can inside the white bowl.'
    logit: 16.667  (base 0.000, delta +16.667)
  [widowx_coke_can_on_ramekin_clean]
    in : 'place the red can upright inside the white bowl'
    out: 'put the red can in the white bowl standing vertical'
    logit: 66.667  (base 50.000, delta +16.667)
  [widowx_spoon_on_towel]
    in : 'put the spoon on the blue cloth'
    out: 'Put the spoon on the blue cloth.'
    logit: 33.333  (base 50.000, delta -16.667)
  [widowx_carrot_on_wheel_clean]
    in : 'place the carrot in the black wheel'
    out: 'Put the carrot in the black wheel.'
    logit: 33.333  (base 33.333, delta +0.000)
  [widowx_stack_cube]
    in : 'Put the green one on top of the yellow one.'
    out: 'Stack the green block on top of the yellow block.'
    logit: 16.667  (base 33.333, delta -16.667)
  [widowx_carrot_on_keyboard_clean]
    in : 'Lay the carrot across the keyboard.'
    out: 'Put the carrot on the keyboard.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_carrot_on_wheel_clean]
    in : 'Move the orange vegetable to the wheel.'
    out: 'Put the carrot on the toy wheel.'
    logit: 16.667  (base 0.000, delta +16.667)
  [widowx_spoon_on_towel]
    in : 'It would be highly appreciated if you could relocate the concave piece of silverware so that it resides atop the woven textile.'
    out: 'Put the spoon on the blue cloth.'
    logit: 33.333  (base 50.000, delta -16.667)
  [widowx_carrot_on_keyboard_clean]
    in : 'Assuming you have grasped the orange root vegetable, please transport it such that it rests atop the device used for typing.'
    out: 'Put the carrot on the keyboard.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_coke_can_on_ramekin_clean]
    in : 'Should you be capable of manipulating the cylindrical cola container, proceed to hoist it and guide it downward until it is securely housed in the ramekin.'
    out: 'Pick up the red cola can and put it in the white bowl.'
    logit: 66.667  (base 0.000, delta +66.667)
  [widowx_carrot_on_plate]
    in : 'You are hereby instructed to commence the relocation of the carrot, terminating said motion only when it makes contact with the plate.'
    out: 'Move the carrot onto the plate.'
    logit: 33.333  (base 16.667, delta +16.667)
  [widowx_carrot_on_plate]
    in : 'Proceed to grasp the carrot and subsequently establish its position atop the dining plate.'
    out: 'Pick up the carrot and place it on the plate.'
    logit: 33.333  (base 33.333, delta +0.000)
  [widowx_carrot_on_plate]
    in : 'The carrot, which is presently situated in its initial starting location, must be lifted and maneuvered until it occupies the plate.'
    out: 'Pick up the carrot and place it on the plate.'
    logit: 33.333  (base 16.667, delta +16.667)
  [widowx_carrot_on_plate]
    in : 'Given the current arrangement of objects, ensure that the carrot is swiftly conveyed to the surface of the plate.'
    out: 'Put the carrot on the plate.'
    logit: 33.333  (base 16.667, delta +16.667)
  [widowx_carrot_on_wheel_clean]
    in : 'The elongated orange taproot must be elevated from its current location and meticulously positioned such that it is fully supported by the wheel.'
    out: 'Pick up the carrot and place it on the wheel.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_stack_cube]
    in : 'Avoiding all other objects, ensure that the highest point of the yellow block becomes the new foundation for the green block.'
    out: 'Stack the green block on top of the yellow block.'
    logit: 16.667  (base 0.000, delta +16.667)
  [widowx_coke_can_on_plate_clean]
    in : 'Assuming the plate is readily accessible, proceed to transport the red tin of cola directly onto its surface.'
    out: 'Move the red can of cola onto the green plate.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_stack_cube]
    in : 'Assuming the yellow block is currently clear, elevate the green block until it can be deposited directly upon its upper surface.'
    out: 'Pick up the green block and stack it on top of the yellow block.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_stack_cube]
    in : 'Without disturbing the surrounding items, it is required that the block of green coloration be made to rest atop its yellow counterpart.'
    out: 'Stack the green cube on top of the yellow cube without moving the other objects.'
    logit: 16.667  (base 0.000, delta +16.667)
  [widowx_put_eggplant_in_basket]
    in : 'Deposit the bulbous purple vegetable directly into the confines of the yellow woven basket.'
    out: 'Put the eggplant into the yellow basket.'
    logit: 100.000  (base 100.000, delta +0.000)
  [widowx_spoon_on_towel]
    in : 'It is hereby requested that the spoon be promptly elevated from its current station and subsequently lowered onto the surface of the towel.'
    out: 'Pick up the spoon and put it on the towel.'
    logit: 33.333  (base 0.000, delta +33.333)