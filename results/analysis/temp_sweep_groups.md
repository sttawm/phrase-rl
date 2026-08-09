# Frozen Qwen3.5-9B — candidate groups by decode temperature

K=16 per group, 5 contexts. v11 trained at **gen-temp 1.0** and ran at
mean dup_frac 0.579 (median 6/16 unique); the frozen model at that same temperature is 14.8/16.

| temp | mean unique of 16 | dup_frac |
|---|---|---|
| 0.7 | 12.6 | 0.212 |
| 1.0 | 14.8 | 0.075 |
| 1.2 | 15.8 | 0.013 |
| 1.5 | 15.8 | 0.013 |


---

## Context 0 — `move the green object from under the heater to the top right outside`

**Trace the model conditions on:**

> <Description of the image> The image shows a toy kitchen countertop. On the black stove, there is a silver pot with a white ladle on the left burner, and a folded orange cloth on the right burner. To the right of the stove, on the white counter space, sits a small green plastic utensil (likely a toy brush or scraper). Directly above this green object is the black robotic gripper. To the far right, a gray sink basin is visible. <Meaning of the instruction in the context of the image> The instruction asks the robot to pick up the green plastic tool (which is currently lying on the white counter directly under the black robotic gripper/arm structure) and relocate it to the upper right corner of the counter surface. Original: green object as many as possible potential replacements: green tool, green scraper, green item, green plastic piece, green brush, green utensil Original: heater as many…

### T = 0.7  —  12/16 unique, dup_frac 0.250

 1. Relocate the green utensil from beneath the black robotic gripper to the upper right corner of the countertop.
 2. Pick up the green tool from beneath the black gripper and place it in the upper right corner of the counter.
 3. Transfer the green utensil from beneath the robotic arm to the upper right corner of the counter.
 4. Transfer the green plastic utensil from beneath the robotic arm to the upper right corner of the counter.
 5. Relocate the green utensil from beneath the robotic gripper to the upper right corner of the counter.
 6. ~~Transfer the green utensil from beneath the robotic arm to the upper right corner of the counter.~~  *(dup)*
 7. Pick up the green utensil from beneath the robotic arm and relocate it to the upper right corner of the countertop.
 8. Relocate the green plastic utensil from beneath the robotic arm to the upper right corner of the counter.
 9. Relocate the green utensil from beneath the robotic arm to the upper right corner of the counter.
10. Relocate the green utensil from beneath the black robotic gripper to the upper right corner of the counter.
11. Relocate the green tool from beneath the robotic arm to the upper right corner of the counter.
12. Transfer the green utensil from beneath the black robotic gripper to the upper right corner of the counter.
13. ~~Relocate the green tool from beneath the robotic arm to the upper right corner of the counter.~~  *(dup)*
14. ~~Relocate the green utensil from beneath the robotic gripper to the upper right corner of the counter.~~  *(dup)*
15. Transfer the green tool from beneath the black gripper to the upper right corner of the counter.
16. ~~Relocate the green utensil from beneath the black robotic gripper to the upper right corner of the counter.~~  *(dup)*

### T = 1.0  —  14/16 unique, dup_frac 0.125

 1. Relocate the green plastic utensil from beneath the black gripper to the upper right corner of the countertop.
 2. Pick up the green tool from under the black gripper and place it in the upper right corner of the counter.
 3. Reposition the green plastic tool from beneath the robotic arm to the upper right corner of the countertop.
 4. Transfer the green plastic utensil from beneath the robotic arm to the upper right corner of the counter.
 5. Relocate the green utensil from beneath the robotic gripper to the upper right corner of the counter.
 6. Transfer the green utensil from beneath the robotic arm to the upper right corner of the counter.
 7. Pick up the green utensil from beneath the robotic gripper and place it in the upper right corner of the counter.
 8. Relocate the green plastic utensil from beneath the robotic arm to the upper right corner of the counter.
 9. Relocate the green utensil from beneath the robotic arm to the upper right corner of the counter.
10. Relocate the green utensil from under the black robotic gripper to the upper right corner of the counter.
11. Relocate the green tool from beneath the robotic arm to the upper right corner of the counter.
12. Transfer the green plastic utensil from beneath the robotic gripper to the upper right corner of the counter.
13. ~~Relocate the green tool from beneath the robotic arm to the upper right corner of the counter.~~  *(dup)*
14. ~~Relocate the green utensil from beneath the robotic gripper to the upper right corner of the counter.~~  *(dup)*
15. Transfer the green tool from beneath the black gripper to the upper right corner of the counter.
16. Relocate the green utensil from beneath the black robotic gripper to the upper right corner of the counter.

### T = 1.2  —  16/16 unique, dup_frac 0.000

 1. Reposition the green plastic tool from beneath the black robotic gripper to the upper right corner of the countertop.
 2. Pick up the green tool from under the black gripper and place it in the upper right corner of the countertop.
 3. Reposition the green plastic tool from under the robotic arm to the upper right corner of the countertop.
 4. Shift the green plastic utensil from beneath the black robotic gripper to the upper right corner of the counter.
 5. Grab the green utensil beneath the black robotic gripper and relocate it to the upper right corner of the countertop.
 6. Transfer the green utensil from beneath the robotic arm to the upper right corner of the counter.
 7. Pick up the green utensil beneath the robotic arm and relocate it to the upper right corner of the counter.
 8. Relocate the green plastic utensil from beneath the robotic arm to the upper right corner of the counter.
 9. Relocate the green utensil from beneath the robotic arm to the upper right corner of the counter.
10. Relocate the green utensil from under the black robotic gripper to the upper right corner of the counter.
11. Move the green tool from under the robotic arm to the upper right corner of the counter.
12. Transfer the green plastic utensil from beneath the robotic gripper to the upper right corner of the counter.
13. Pick up the green tool from beneath the robotic arm and move it to the upper right corner of the counter.
14. Relocate the green utensil from beneath the robotic gripper to the upper right corner of the counter.
15. Transfer the green tool from beneath the black gripper to the upper right corner of the counter.
16. Relocate the green utensil from beneath the black robotic gripper to the upper right corner of the counter.

### T = 1.5  —  16/16 unique, dup_frac 0.000

 1. Reposition the green plastic tool from beneath the black robotic gripper to the upper-right corner of the countertop.
 2. Pick up the green tool from under the black gripper and place it in the upper right corner of the countertop.
 3. Reposition the green plastic tool from under the robotic arm to the upper right corner of the countertop.
 4. Shift the green plastic brush from beneath the robotic arm to the upper right corner of the counter.
 5. Grab the green utensil beneath the black robotic gripper and relocate it to the upper right corner of the countertop.
 6. Transfer the green utensil from beneath the robotic arm to the upper-right corner of the counter.
 7. Pick up the green utensil beneath the robotic arm and reposition it to the upper right corner of the countertop.
 8. Relocate the green plastic utensil from beneath the robotic arm to the upper right corner of the counter.
 9. Relocate the green utensil from beneath the robotic arm to the upper right section of the counter.
10. Relocate the green utensil from under the black robotic gripper to the upper right corner of the counter.
11. Move the green tool from under the robotic arm to the upper right corner of the counter.
12. Translate this action: "Pick up the green toy utensil from beneath the black robotic arm and move it to the far top-right section of the countertop."
13. Pick up the green tool from beneath the robotic arm and move it to the upper right corner of the counter.
14. Relocate the green utensil from beneath the robotic gripper to the upper right corner of the counter.
15. Transfer the green tool from beneath the black robotic gripper to the upper right corner of the counter.
16. Relocate the green utensil from beneath the black robotic gripper to the upper right corner of the counter.


---

## Context 1 — `put sushi in pot cardboard fence`

**Trace the model conditions on:**

> The image shows a robotic gripper positioned over a wooden workspace enclosed by a cardboard boundary (cardboard fence). Inside this area, there are several toy items: a yellow banana, a green pear-shaped toy, a red chili pepper, an orange translucent plastic pot, and a slice of toy sushi representing salmon nigiri (white rice topped with orange fish). The instruction asks the robot to grab the orange and white sushi slice and place it inside the orange translucent pot that is located within the cardboard fenced area. Original: sushi as many as possible potential replacements: salmon sushi, sushi slice, sushi toy, orange and white sushi, sushi piece, toy food Original: pot as many as possible potential replacements: orange pot, orange cup, orange container, translucent bowl, plastic pot, orange dish Original: cardboard fence as many as possible potential replacements: cardboard tray, fen…

### T = 0.7  —  9/16 unique, dup_frac 0.438

 1. Place the orange and white sushi slice into the orange translucent pot inside the cardboard fence.
 2. Place the orange and white sushi slice inside the orange translucent pot within the cardboard fence.
 3. Place the orange and white sushi slice into the orange translucent pot within the cardboard fence.
 4. Place the orange and white sushi slice into the translucent orange pot within the cardboard fence.
 5. ~~Place the orange and white sushi slice into the orange translucent pot within the cardboard fence.~~  *(dup)*
 6. Place the orange and white sushi slice inside the orange translucent pot within the cardboard fenced area.
 7. Place the orange and white sushi slice in the orange translucent pot within the cardboard fenced area.
 8. ~~Place the orange and white sushi slice inside the orange translucent pot within the cardboard fenced area.~~  *(dup)*
 9. ~~Place the orange and white sushi slice inside the orange translucent pot within the cardboard fence.~~  *(dup)*
10. Transfer the orange and white sushi slice to the orange translucent pot within the cardboard enclosure.
11. ~~Place the orange and white sushi slice into the orange translucent pot within the cardboard fence.~~  *(dup)*
12. Transfer the orange and white sushi slice to the orange translucent pot inside the cardboard fence.
13. ~~Place the orange and white sushi slice inside the orange translucent pot within the cardboard fenced area.~~  *(dup)*
14. Transfer the orange and white sushi toy to the translucent orange pot within the cardboard fenced area.
15. ~~Place the orange and white sushi slice inside the orange translucent pot within the cardboard fence.~~  *(dup)*
16. ~~Place the orange and white sushi slice inside the orange translucent pot within the cardboard fence.~~  *(dup)*

### T = 1.0  —  15/16 unique, dup_frac 0.062

 1. Place the orange and white sushi slice into the orange translucent pot inside the cardboard fence.
 2. Place the orange and white sushi slice inside the orange translucent pot within the cardboard fence.
 3. Place the orange and white sushi slice into the orange translucent pot within the cardboard fence.
 4. place the orange and white sushi slice into the translucent orange plastic pot within the cardboard fence
 5. Place the orange and white sushi toy into the orange translucent plastic pot inside the cardboard fenced area.
 6. Place the orange and white sushi slice inside the orange translucent pot within the cardboard boundary.
 7. Transfer the orange and white sushi slice into the orange translucent pot within the cardboard fenced area.
 8. Please place the orange and white sushi slice inside the orange translucent pot within the cardboard fenced area.
 9. Insert the orange and white sushi toy into the orange translucent plastic pot within the cardboard fenced area.
10. Transfer the orange and white sushi slice to the orange translucent pot within the cardboard enclosure.
11. Place the orange and white sushi toy into the orange translucent pot within the cardboard enclosure.
12. place the orange and white sushi slice in the orange pot inside the cardboard fence
13. Place the orange and white sushi slice inside the orange translucent pot within the cardboard fenced area.
14. Transfer the orange and white sushi toy to the translucent orange pot within the cardboard boundary.
15. Place the orange and white sushi toy inside the translucent orange plastic pot located within the cardboard boundary.
16. ~~Place the orange and white sushi slice inside the orange translucent pot within the cardboard fence.~~  *(dup)*

### T = 1.2  —  15/16 unique, dup_frac 0.062

 1. Place the orange and white sushi slice into the orange translucent pot inside the cardboard fence.
 2. Place the orange and white sushi slice inside the orange translucent pot within the cardboard fence.
 3. Place the orange and white sushi slice into the orange translucent pot within the cardboard fence.
 4. place the orange and white sushi slice into the translucent orange plastic pot within the cardboard fence
 5. Place the orange and white sushi toy into the orange translucent plastic pot inside the cardboard fenced area.
 6. Place the orange and white sushi slice inside the orange translucent pot within the cardboard boundary.
 7. Transfer the orange and white sushi slice into the orange translucent pot within the cardboard fenced area.
 8. Please place the orange and white sushi slice inside the orange translucent pot within the cardboard fenced area.
 9. Insert the orange and white sushi toy into the orange translucent plastic pot within the cardboard enclosed area.
10. Transfer the orange and white sushi slice to the orange translucent pot within the cardboard enclosure.
11. Place the orange and white sushi toy into the orange translucent pot within the cardboard enclosure.
12. place the orange and white sushi slice in the orange pot inside the cardboard fence
13. Place the orange and white sushi slice inside the orange translucent pot within the cardboard fenced area.
14. Transfer the orange and white sushi toy to the translucent orange pot within the cardboard boundary.
15. Place the orange and white sushi toy inside the translucent orange plastic pot located within the cardboard boundary.
16. ~~Place the orange and white sushi slice inside the orange translucent pot within the cardboard fence.~~  *(dup)*

### T = 1.5  —  16/16 unique, dup_frac 0.000

 1. Place the orange and white sushi slice into the orange translucent pot inside the cardboard fence.
 2. Place the orange and white sushi slice in the orange plastic pot inside the cardboard fence.
 3. Place the orange and white sushi slice into the orange translucent pot within the cardboard fence.
 4. place the orange and white sushi slice into the translucent orange plastic pot within the cardboard fence
 5. Place the orange and white sushi toy into the orange translucent plastic pot inside the cardboard fenced area.
 6. Place the orange and white sushi slice inside the orange plastic pot within the cardboard boundary.
 7. Transfer the toy sushi slice and insert it into the orange translucent pot inside the cardboard fence.
 8. Please place the orange and white sushi slice inside the orange translucent pot within the cardboard fenced area.
 9. Insert the orange and white sushi toy into the orange plastic container within the cardboard fence.
10. Transfer the orange and white sushi slice to the orange translucent pot within the cardboard enclosure.
11. Place the orange and white toy sushi into the orange translucent pot within the cardboard enclosure.
12. place the orange and white sushi slice in the orange pot inside the cardboard fence
13. Place the orange and white sushi slice inside the orange translucent pot within the cardboard fenced area.
14. Transfer the toy nigiri sushi slice to the translucent orange pot within the cardboard boundary.
15. Place the orange and white sushi toy inside the translucent orange plastic pot located within the cardboard boundary.
16. place the orange and white sushi slice inside the orange translucent pot within the cardboard fence


---

## Context 2 — `dyeing cloth`

**Trace the model conditions on:**

> **Description of the image:** A robotic arm gripper is holding a blue microfiber cloth (or rag) on top of a black toy stovetop burner, which is situated next to a small gray toy sink. **Meaning of the instruction in the context of the image:** The original instruction "dyeing cloth" appears to be a typographical error for "drying cloth" or "wiping with cloth." In the context of the image, the robot is holding a blue cloth on a toy stovetop, likely simulating the act of drying the cloth on a heat source or using the cloth to wipe/dry the stovetop surface. **Original: cloth** as many as possible potential replacements: rag, towel, microfiber cloth, fabric, wipe **Original: dyeing (drying)** as many as possible potential replacements: drying, wiping, cleaning, placing, heating **Original: (none in original)** as many as possible potential replacements: blue, toy, black, wet **Original: (non…

### T = 0.7  —  15/16 unique, dup_frac 0.062

 1. Place the blue microfiber cloth on the toy stovetop burner.
 2. wipe stovetop with cloth
 3. Dry the blue cloth on the toy stovetop.
 4. Place the blue microfiber cloth on the stovetop to dry.
 5. Dry the blue microfiber cloth on the toy stovetop.
 6. Place the blue microfiber cloth on the black toy stovetop burner.
 7. Place the blue microfiber cloth on the black toy stovetop.
 8. Dry the cloth on the toy stove.
 9. dry the cloth on the stovetop
10. Place the blue cloth on the toy stovetop.
11. dry the cloth
12. Wipe the toy stovetop with the blue microfiber cloth.
13. dry cloth
14. dry cloth on the toy stovetop
15. dry cloth on stovetop
16. ~~dry the blue cloth on the toy stovetop~~  *(dup)*

### T = 1.0  —  15/16 unique, dup_frac 0.062

 1. Place the blue cloth on the stovetop burner to dry.
 2. wipe stovetop with cloth
 3. Dry the blue cloth on the toy stovetop.
 4. Place the blue microfiber cloth on the stovetop to dry.
 5. Dry the blue microfiber cloth on the toy stovetop.
 6. Place the cloth on the burner to dry it.
 7. Place the blue microfiber cloth on the black toy stovetop.
 8. Dry the cloth on the toy stove.
 9. dry the cloth on the stove
10. Place the blue cloth on the toy stovetop.
11. ~~dry the blue microfiber cloth on the toy stovetop~~  *(dup)*
12. Wipe the toy stovetop with the blue rag.
13. dry cloth on stovetop
14. dry cloth on the toy stovetop
15. Drying the cloth on the toy stovetop.
16. Wiping stovetop with cloth

### T = 1.2  —  16/16 unique, dup_frac 0.000

 1. Place the blue cloth on the stovetop burner to dry.
 2. wipe stovetop with cloth
 3. Dry the blue cloth on the toy stovetop.
 4. Place the blue microfiber cloth on the stovetop to dry.
 5. dry the blue cloth
 6. Place the cloth on the burner to dry it.
 7. dry the cloth
 8. Dry the cloth on the toy stove.
 9. place drying cloth on stovetop
10. Place the blue cloth on the toy stovetop.
11. dry the blue rag
12. Wipe the toy stovetop with the blue rag.
13. drying the cloth
14. dry cloth on stove
15. Drying the cloth on the toy stovetop.
16. Wiping stovetop with cloth

### T = 1.5  —  16/16 unique, dup_frac 0.000

 1. Place the blue cloth on the stovetop burner to dry.
 2. wipe stovetop with cloth
 3. Dry the blue cloth on the stove
 4. Place the blue towel on the stovetop.
 5. dry the blue cloth
 6. Place the cloth on the burner to dry it.
 7. dry the cloth
 8. Dry the cloth with the gripper.
 9. place drying cloth on black stove
10. Place the blue cloth on the toy stovetop.
11. dry the blue rag
12. Wipe the toy stovetop with the blue rag
13. drying the cloth
14. dry cloth on stove
15. Drape the cloth on the toy stovetop burner.
16. Wiping stovetop with cloth


---

## Context 3 — `Take the orange cloth and put it inside the washing machine.`

**Trace the model conditions on:**

> <Description of the image> The image shows a robotic arm gripper grasping an orange cloth inside the open drum of a front-loading washing machine. Below the washing machine, on a wooden platform, there is a small white plastic basket containing a yellow folded cloth. <Meaning of the instruction in the context of the image> The instruction directs the robot to grasp the orange cloth and place it inside the washing drum of the washing machine. Original: cloth as many as possible potential replacements: rag, fabric, towel, wipe, textile, laundry item Original: washing machine as many as possible potential replacements: washer, laundry machine, washing appliance, drum, appliance Original: Take as many as possible potential replacements: pick up, grab, retrieve, grasp, collect, lift Original: put as many as possible potential replacements: place, deposit, set, insert, drop, load, transfer, po…

### T = 0.7  —  14/16 unique, dup_frac 0.125

 1. Grasp the orange-colored fabric and insert it into the washing machine.
 2. Grasp the orange-colored fabric and insert it into the washer drum.
 3. Grasp the orange-colored fabric and insert it into the washing machine drum.
 4. Grab the orange fabric and insert it into the washer.
 5. Grasp the orange-colored textile and deposit it into the drum of the washing appliance.
 6. Grab the orange-colored fabric and load it into the washer.
 7. Grasp the orange fabric and place it into the washer.
 8. Grasp the orange fabric and insert it into the washing machine drum.
 9. ~~Grasp the orange-colored fabric and insert it into the washing machine drum.~~  *(dup)*
10. Grasp the orange-colored fabric and deposit it into the drum of the washer.
11. Grab the orange-colored fabric and insert it into the washer.
12. Grasp the orange fabric and insert it into the washer.
13. Grab the orange fabric and place it into the washer.
14. Grasp the orange-colored fabric and place it into the washing machine drum.
15. ~~Grasp the orange fabric and insert it into the washer.~~  *(dup)*
16. Grasp the orange fabric and deposit it into the washing machine drum.

### T = 1.0  —  16/16 unique, dup_frac 0.000

 1. Grasp the orange-colored fabric and insert it into the washing machine.
 2. Grasp the orange-colored fabric and insert it into the washer drum.
 3. Pick up the orange-colored fabric and insert it into the washing drum.
 4. Grab the orange fabric and insert it into the washer.
 5. Grasp the orange-colored textile and deposit it into the drum of the washing appliance.
 6. Grab the bright orange rag and load it into the washer.
 7. Grasp the orange fabric and place it into the washer.
 8. Pick up the orange fabric and insert it into the washing machine.
 9. Grasp the orange-colored fabric and insert it into the washing machine drum.
10. Grasp the orange-colored fabric and deposit it within the drum of the laundry machine.
11. Grab the orange-colored fabric and insert it into the washer.
12. Retrieve the orange-colored fabric and deposit it into the drum of the washing appliance.
13. Grab the orange fabric and place it within the washer.
14. Pick up the orange-colored rag and place it into the washer.
15. Grasp the orange fabric and insert it into the washer drum.
16. Grasp the orange fabric and deposit it into the washing machine drum.

### T = 1.2  —  16/16 unique, dup_frac 0.000

 1. Grasp the orange-colored laundry item and place it into the washer.
 2. Grasp the orange-colored fabric and insert it into the washer drum.
 3. Pick up the orange-colored rag and insert it into the washing drum.
 4. Grab the orange fabric and insert it into the washer.
 5. Grasp the orange cloth and load it into the washer drum.
 6. Grab the bright orange rag and load it into the washer.
 7. Grasp the orange textile and deposit it into the washer's drum.
 8. Pick up the orange fabric and insert it into the washing appliance.
 9. Grasp the orange-colored fabric and insert it into the washing machine drum.
10. Grasp the orange-colored fabric and deposit it within the drum of the laundry machine.
11. Grab the orange-colored fabric and insert it into the washer.
12. Retrieve the orange-colored fabric and deposit it into the drum of the washing appliance.
13. Grab the orange fabric and place it within the washer.
14. Pick up the orange-colored rag and place it into the washer.
15. Grasp the orange fabric and insert it within the washer's drum.
16. Grasp the orange fabric and deposit it within the washing machine drum.

### T = 1.5  —  16/16 unique, dup_frac 0.000

 1. Grasp the orange-colored laundry item and place it into the washer.
 2. Grasp the orange cloth and deposit it into the washing machine drum.
 3. Pick up the orange-colored rag and insert it into the washing drum.
 4. Grab the orange fabric and lift it into the washer.
 5. Grasp the orange cloth within the washing machine and transfer it as many as possible.
 6. Grab the bright orange rag and load it into the washer.
 7. Grasp the orange textile and deposit it within the washer's drum.
 8. Pick up the orange fabric and insert it into the washing appliance.
 9. Grasp the bright orange fabric and insert it into the washing machine drum.
10. Grasp the orange-colored fabric and deposit it within the drum of the laundry machine.
11. Grab the orange-colored fabric and insert it into the washer.
12. Retrieve the orange-colored fabric and deposit it into the drum of the washing appliance.
13. Grab the orange fabric and place it within the washer.
14. Pick up the orange-colored rag and place it into the washer.
15. Grasp the orange fabric and insert it within the washing machine's drum.
16. Grasp the bright orange laundry item and insert it into the washer.


---

## Context 4 — `end effector reaching corn`

**Trace the model conditions on:**

> <Description of the image> The image shows a toy kitchen stove setup with a robotic arm and gripper positioned above it. On the stove, there is a silver pot on the right burner, a green-lidded container on the left burner, a yellow-handled toy knife on the left side, and a yellow plastic toy corn sitting on the front control panel between the knobs. <Meaning of the instruction in the context of the image> The instruction directs the robotic gripper (end effector) to move from its current position and reach towards or touch the yellow toy corn located on the stove's front console. Original: end effector as many as possible potential replacements: gripper, robotic hand, robotic claw, robot arm, claw, hand, gripper head Original: reaching as many as possible potential replacements: moving to, approaching, touching, reaching out to, going to, extending to, hovering over, guiding to Original:…

### T = 0.7  —  13/16 unique, dup_frac 0.188

 1. gripper approaching yellow toy corn
 2. The robotic gripper moving to the yellow toy corn.
 3. robotic hand approaching toy corn
 4. Move the robotic gripper to reach the yellow toy corn.
 5. robotic hand approaching yellow toy corn
 6. gripper moving to toy corn
 7. The robotic gripper should reach the toy corn.
 8. move the robotic gripper to the toy corn
 9. ~~robotic hand approaching toy corn~~  *(dup)*
10. The robotic gripper should approach the yellow toy corn.
11. gripper reaching out to toy corn
12. ~~robotic hand approaching toy corn~~  *(dup)*
13. Move the robotic gripper to reach the toy corn.
14. robotic arm approaching toy corn
15. robotic gripper approaching toy corn
16. ~~Move the robotic gripper to the toy corn.~~  *(dup)*

### T = 1.0  —  14/16 unique, dup_frac 0.125

 1. gripper approaching yellow toy corn
 2. The robotic gripper moving to the yellow toy corn.
 3. robotic hand approaching toy corn
 4. Move the robotic hand toward the toy corn.
 5. robotic hand approaching yellow toy corn
 6. gripper moving to toy corn
 7. The robotic gripper should reach the toy corn.
 8. move robotic hand to yellow toy corn
 9. ~~robotic hand approaching toy corn~~  *(dup)*
10. grripper moving to toy corn
11. gripper reaching out to toy corn
12. ~~robotic hand approaching toy corn~~  *(dup)*
13. Move the robotic gripper to reach the toy corn.
14. robotic arm approaching toy corn
15. robotic gripper approaching toy corn
16. Robot arm approaching the toy corn

### T = 1.2  —  16/16 unique, dup_frac 0.000

 1. gripper approaching yellow toy corn
 2. The robotic gripper moving to the yellow toy corn on the stove's control panel.
 3. Robotic hand to reach yellow corn
 4. Move the robotic arm to touch the toy corn.
 5. robotic hand approaching yellow toy corn
 6. gripper moving to toy corn
 7. The robotic gripper should reach the toy corn.
 8. move robotic hand to yellow toy corn
 9. Extend the robotic gripper towards the toy corn.
10. grripper approaching yellow toy corn
11. gripper reaching out to toy corn
12. robotic hand approaching toy corn
13. Move the robotic gripper to reach the toy corn.
14. robotic arm approaching toy corn
15. robotic gripper approaching toy corn
16. Robot gripper reaching toy corn

### T = 1.5  —  15/16 unique, dup_frac 0.062

 1. gripper approaching yellow toy corn
 2. The robotic gripper must move towards and touch the yellow toy corn located on the front panel of the stove.
 3. Robotic hand to toy corn
 4. The gripper moves toward the toy corn.
 5. robotic hand approaching yellow toy corn
 6. gripper moving to toy corn
 7. The robotic gripper should reach the toy corn.
 8. move to the toy corn
 9. Extend the robotic gripper towards the yellow toy corn.
10. grripper approaching yellow toy corn
11. gripper reaching out to toy corn
12. ~~robotic hand to toy corn~~  *(dup)*
13. Robot arm extending gripper towards toy corn on front panel
14. robotic arm approaching toy corn
15. robotic gripper approaching toy corn
16. Extend gripper to touch toy corn on the front panel.
