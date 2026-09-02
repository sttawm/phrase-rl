# Previous rulebook: measured performance

Whole rulebook applied: mean proxy logit 0.288 vs 0.229 for the unrephrased instruction (+0.059; higher is better, not a probability).

## Sample rewrites from the whole rulebook
(logit: higher is better; delta = rewrite - base, positive means
the rewrite beat leaving the instruction alone)

  [widowx_coke_can_on_plate_clean]
    in : 'put the coke down on the plate'
    out: 'put the coke down on the plate'
    logit: 83.333  (base 83.333, delta +0.000)
  [widowx_carrot_on_wheel_clean]
    in : 'Transfer the carrot over to the wheel.'
    out: 'Transfer the carrot over to the wheel.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_carrot_on_plate]
    in : 'put the carrot on top of the plate'
    out: 'put the carrot on top of the plate'
    logit: 33.333  (base 33.333, delta +0.000)
  [widowx_carrot_on_keyboard_clean]
    in : 'set the carrot on the keyboard'
    out: 'set the carrot on the keyboard'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_carrot_on_wheel_clean]
    in : 'set the carrot on the wheel'
    out: 'set the carrot on the wheel'
    logit: 16.667  (base 16.667, delta +0.000)
  [widowx_put_eggplant_in_basket]
    in : 'Slide the eggplant into the yellow basket.'
    out: 'Slide the eggplant into the yellow basket.'
    logit: 83.333  (base 83.333, delta +0.000)
  [widowx_coke_can_on_plate_clean]
    in : 'place coke can on plate'
    out: 'place coke can on plate'
    logit: 50.000  (base 50.000, delta +0.000)
  [widowx_carrot_on_plate]
    in : 'Please shift the carrot onto the plate.'
    out: 'put the carrot onto the plate'
    logit: 16.667  (base 16.667, delta +0.000)
  [widowx_carrot_on_keyboard_clean]
    in : 'Pick up the carrot and set it down on the keyboard.'
    out: 'Pick up the carrot and set it down on the keyboard.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_carrot_on_keyboard_clean]
    in : 'place the carrot across the black keyboard'
    out: 'place the carrot across the black keyboard'
    logit: 16.667  (base 0.000, delta +16.667)
  [widowx_coke_can_on_ramekin_clean]
    in : 'Pick up the red cola can and place it upright inside the white bowl.'
    out: 'Pick up the red cola can and place it upright inside the white bowl.'
    logit: 50.000  (base 50.000, delta +0.000)
  [widowx_coke_can_on_ramekin_clean]
    in : 'place the cola can standing up inside the white bowl'
    out: 'place the cola can standing up inside the white bowl'
    logit: 16.667  (base 50.000, delta -33.333)
  [widowx_coke_can_on_plate_clean]
    in : 'Grab that red aluminum can and set it on the plate.'
    out: 'Grab that red aluminum can and set it on the plate.'
    logit: 66.667  (base 66.667, delta +0.000)
  [widowx_spoon_on_towel]
    in : 'Could you grab that spoon and set it on the cloth for me?'
    out: 'pick up that spoon and put it on the cloth'
    logit: 50.000  (base 50.000, delta +0.000)
  [widowx_stack_cube]
    in : 'Stack the green on the yellow.'
    out: 'Stack the green on the yellow.'
    logit: 50.000  (base 50.000, delta +0.000)
  [widowx_stack_cube]
    in : 'green block on yellow block'
    out: 'green block on yellow block'
    logit: 16.667  (base 16.667, delta +0.000)
  [widowx_stack_cube]
    in : 'Can you place the green block onto the yellow block for me?'
    out: 'place the green block onto the yellow block'
    logit: 0.000  (base 33.333, delta -33.333)
  [widowx_coke_can_on_plate_clean]
    in : 'put the coke can on the dish'
    out: 'put the coke can on the dish'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_stack_cube]
    in : 'put the green cube on the yellow one'
    out: 'put the green cube on the yellow one'
    logit: 33.333  (base 33.333, delta +0.000)
  [widowx_coke_can_on_ramekin_clean]
    in : 'place the cola can inside the white bowl'
    out: 'place the cola can inside the white bowl'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_coke_can_on_ramekin_clean]
    in : 'place the red can upright inside the white bowl'
    out: 'place the red can upright inside the white bowl'
    logit: 66.667  (base 50.000, delta +16.667)
  [widowx_spoon_on_towel]
    in : 'put the spoon on the blue cloth'
    out: 'put the spoon on the blue cloth'
    logit: 33.333  (base 50.000, delta -16.667)
  [widowx_carrot_on_wheel_clean]
    in : 'place the carrot in the black wheel'
    out: 'place the carrot in the black wheel'
    logit: 33.333  (base 33.333, delta +0.000)
  [widowx_stack_cube]
    in : 'Put the green one on top of the yellow one.'
    out: 'Put the green one on top of the yellow one.'
    logit: 33.333  (base 33.333, delta +0.000)
  [widowx_carrot_on_keyboard_clean]
    in : 'Lay the carrot across the keyboard.'
    out: 'Lay the carrot across the keyboard.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_carrot_on_wheel_clean]
    in : 'Move the orange vegetable to the wheel.'
    out: 'Move the orange carrot to the wheel.'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_spoon_on_towel]
    in : 'It would be highly appreciated if you could relocate the concave piece of silverware so that it resides atop the woven textile.'
    out: 'put the spoon on the woven cloth'
    logit: 66.667  (base 50.000, delta +16.667)
  [widowx_carrot_on_keyboard_clean]
    in : 'Assuming you have grasped the orange root vegetable, please transport it such that it rests atop the device used for typing.'
    out: 'put the orange carrot on the keyboard'
    logit: 33.333  (base 0.000, delta +33.333)
  [widowx_coke_can_on_ramekin_clean]
    in : 'Should you be capable of manipulating the cylindrical cola container, proceed to hoist it and guide it downward until it is securely housed in the ramekin.'
    out: 'pick up the can and put it down in the ramekin'
    logit: 0.000  (base 0.000, delta +0.000)
  [widowx_carrot_on_plate]
    in : 'You are hereby instructed to commence the relocation of the carrot, terminating said motion only when it makes contact with the plate.'
    out: 'put the carrot on the plate'
    logit: 16.667  (base 16.667, delta +0.000)
  [widowx_carrot_on_plate]
    in : 'Proceed to grasp the carrot and subsequently establish its position atop the dining plate.'
    out: 'pick up the carrot and put it on the dining plate'
    logit: 16.667  (base 33.333, delta -16.667)
  [widowx_carrot_on_plate]
    in : 'The carrot, which is presently situated in its initial starting location, must be lifted and maneuvered until it occupies the plate.'
    out: 'pick up the carrot and put it on the plate'
    logit: 33.333  (base 16.667, delta +16.667)
  [widowx_carrot_on_plate]
    in : 'Given the current arrangement of objects, ensure that the carrot is swiftly conveyed to the surface of the plate.'
    out: 'put the carrot on the surface of the plate'
    logit: 16.667  (base 16.667, delta +0.000)
  [widowx_carrot_on_wheel_clean]
    in : 'The elongated orange taproot must be elevated from its current location and meticulously positioned such that it is fully supported by the wheel.'
    out: 'pick up the orange carrot and put it on the wheel'
    logit: 33.333  (base 0.000, delta +33.333)
  [widowx_stack_cube]
    in : 'Avoiding all other objects, ensure that the highest point of the yellow block becomes the new foundation for the green block.'
    out: 'put the green block on the yellow block'
    logit: 33.333  (base 0.000, delta +33.333)
  [widowx_coke_can_on_plate_clean]
    in : 'Assuming the plate is readily accessible, proceed to transport the red tin of cola directly onto its surface.'
    out: 'put the red can on the plate'
    logit: 66.667  (base 0.000, delta +66.667)
  [widowx_stack_cube]
    in : 'Assuming the yellow block is currently clear, elevate the green block until it can be deposited directly upon its upper surface.'
    out: 'pick up the green block and put it on the yellow block'
    logit: 16.667  (base 0.000, delta +16.667)
  [widowx_stack_cube]
    in : 'Without disturbing the surrounding items, it is required that the block of green coloration be made to rest atop its yellow counterpart.'
    out: 'put the green block on the yellow block'
    logit: 33.333  (base 0.000, delta +33.333)
  [widowx_put_eggplant_in_basket]
    in : 'Deposit the bulbous purple vegetable directly into the confines of the yellow woven basket.'
    out: 'put the purple eggplant into the yellow basket'
    logit: 83.333  (base 100.000, delta -16.667)
  [widowx_spoon_on_towel]
    in : 'It is hereby requested that the spoon be promptly elevated from its current station and subsequently lowered onto the surface of the towel.'
    out: 'pick up the spoon and put it on the towel'
    logit: 16.667  (base 0.000, delta +16.667)