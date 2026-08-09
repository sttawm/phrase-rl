# Corpus vocabulary — exact membership test

17297 instructions, 2192 distinct tokens, 555 above a
floor of 5 occurrences (rarer words excluded as noise). Tokenised as
lowercase runs of letters and apostrophes.

A word is corpus-present if and only if it appears below. Presence licenses a
token; it never makes a token good — frequency is not outcome.

**Common (n>=50):** the 39715, of 10330, to 7926, move 6813, put 4895, on 4692, left 4637, right 4428, table 4245, cloth 4102, pot 3709, and 3560, top 3403, side 3032, blue 2742, it 2460, from 2318, place 2176, yellow 2153, silver 2107, bottom 2058, in 1996, burner 1941, red 1758, green 1696, object 1553, spoon 1492, moved 1269, towel 1218, corner 1190, orange 1171, take 1053, stove 994, upper 958, pick 881, block 868, middle 827, drawer 803, front 783, up 762, edge 714, lower 713, cube 713, can 694, white 666, pan 661, fork 624, spatula 587, bowl 507, purple 470, between 463, lid 451, center 433, a 432, inside 413, brush 409, knife 405, arch 394, into 388, out 353, above 329, remove 323, thing 320, fold 317, basket 305, behind 303, near 296, toy 296, unfold 277, with 270, at 265, cylinder 261, pepper 260, back 254, mushroom 253, rectangular 252, next 245, banana 236, sushi 231, tower 227, sink 220, rectangle 215, metal 213, placed 207, napkin 199, counter 194, corn 190, figure 180, below 178, vessel 163, microwave 161, burners 158, part 145, brown 145, off 142, onto 133, other 133, machine 127, two 126, la 124, beside 123, towards 123, carrot 122, far 121, cup 121, down 121, moves 117, over 109, slide 109, cover 106, box 105, hole 103, strawberry 102, washing 100, push 98, bottle 97, black 94, ball 92, item 91, pink 91, cans 88, picked 88, broccoli 88, bell 87, just 87, potato 86, moving 85, took 83, vegetable 83, eggplant 82, violet 81, egg 80, triangle 78, centre 78, wooden 76, le 76, chicken 75, container 74, cheese 74, de 72, close 69, tin 66, is 66, tomato 66, bread 65, another 65, colander 64, piece 63, removed 63, rag 62, utensil 61, under 60, so 60, open 59, duck 58, plastic 58, gray 57, robot 57, or 56, croissant 56, cooker 56, ladle 55, plate 55, steel 54, keep 54, square 54, wall 53, cucumber 52, dish 50, clothe 50

**Present (n=5-49):** across, against, al, along, amarillo, an, animal, anything, apple, arc, area, arm, aside, au, avocado, away, azul, backwards, bag, banner, bar, bas, basin, baster, bear, before, beige, beneath, berry, besides, big, bit, blanket, blender, bleu, blocks, blueberry, board, bord, botton, bowel, brick, bring, brinjal, bucket, bun, buner, bunner, bunny, but, by, cabinet, cake, canned, cap, capsicum, cauliflower, change, chess, chili, chocolate, circle, circular, closed, closer, closes, clothes, clothing, cloths, cob, coloca, color, cone, cooking, cooktop, cot, covered, cream, cubes, cuboid, cupcake, cutting, cylindrical, d, da, dans, dark, del, dentro, derecha, derecho, desk, diagonally, did, direction, directly, do, dog, doll, door, downward, drag, drainer, draw, droite, drop, drumstick, dryer, du, e, el, elephant, empty, en, end, estufa, et, fabric, facing, fish, flip, floor, folded, folding, folds, folk, food, for, form, forward, four, fruit, frying, further, gas, gauche, gaveta, get, glass, goods, grab, grabbed, grabs, grape, grapes, grater, grey, hacia, half, hand, handkerchief, handle, handled, has, haut, he, heater, hexagon, hob, hold, hot, hotdog, ice, image, images, induction, inferior, infront, it's, items, its, izquierda, izquierdo, jar, kitchen, l, label, laddle, lado, lamba, lata, laundry, leave, leftside, leg, lemon, let, lift, light, lime, little, lo, loaded, long, maize, make, mango, marmite, maroon, mat, measuring, meat, mesa, metallic, mettre, mid, monkey, moove, mouse, mover, moveu, mueve, n, no, not, nothing, o, objects, objet, objeto, olla, one, onion, ontop, opened, opening, opens, opposite, outside, oven, p, pack, para, parallelepiped, parte, pawn, pear, peeler, picking, pickle, picks, pickup, pile, pineapple, placer, places, placing, plant, plata, plateada, plier, plus, plush, po, pointed, poner, position, pots, prism, progresses, pull, pulled, pumpkin, puppy, pushed, pushing, puting, puts, putting, pyramid, quemador, rabbit, rack, rear, removing, rieur, righ, rigth, ring, roll, rotate, round, s, salmon, salt, same, sauce, saucepan, sausage, scoop, scrubber, sequence, set, shaker, shape, shaped, shelf, shrimp, slice, slightly, sliver, small, soap, something, soup, space, spatule, sphere, sponge, spoons, spot, squash, stack, stainless, stand, standing, stick, stoves, stovetop, straight, strainer, stuff, stuffed, stuffedduck, sup, superior, sur, surface, sweet, switch, t, table's, tablecloth, taken, takes, taking, tall, te, teddy, that, then, thigh, thin, things, this, tiger, tiroir, tissu, toma, touch, touched, touches, touching, toward, transfer, transferred, transparent, tray, triangular, turn, un, unfolded, unfolding, unfolds, upright, upside, upthe, upward, using, verde, vers, video, vissel, was, wash, washer, wedge, which, wing, wipe, wok, wood, y, yellon, yelow

---

## Shape of the corpus

Instructions in the sample are typically concise, generally ranging from 8 to 15 words in length. The dominant sentence form is the imperative command, though there is a notable minority of past-tense declarative phrasing (e.g., "moved" appears 1269 times, "took" 83 times, "placed" 207 times) and third-person present tense ("moves" 117, "takes" 38). 

Instructions overwhelmingly begin with action verbs. "Move" (6813 occurrences) and "put" (4895) are the most frequent anchors, followed by "place" (2176), "take" (1053), "pick" (881), "remove" (323), "fold" (317), and "unfold" (277). 

Definite articles are ubiquitous, while indefinite articles are rare. The word "the" appears 39,715 times, averaging approximately 2.3 occurrences per instruction across the 17,297-instruction corpus. In contrast, "a" appears only 432 times, and "an" falls into the rare 5–49 occurrence tier. 

The vast majority of instructions describe a single step. However, the conjunction "and" appears 3,560 times, suggesting that roughly 20% of the corpus contains multi-step or compound actions (e.g., "Pick up the spoon from the piece of cloth and place it near the steel pan.").

## Register

Objects are primarily described using basic category nouns, colors, and highly specific positional references. 

*   **Category:** Generic household and geometric terms are standard. Common examples include "cloth" (4102), "pot" (3709), "spoon" (1492), "towel" (1218), and "block" (868). Vague categorical terms are also frequent, such as "object" (1553) and "thing" (320).
*   **Color:** Color is the most frequent descriptive modifier. "Blue" (2742), "yellow" (2153), "red" (1758), "green" (1696), and "orange" (1171) are all highly common.
*   **Material:** Material descriptors are present but less frequent than colors, with "silver" being a major exception (2107 occurrences, frequently paired with "pot" or "pan"). Other materials include "metal" (213), "wooden" (76), "plastic" (58), and "steel" (54).
*   **Position:** Positional and spatial descriptions are extremely precise and frequent. "Left" (4637) and "right" (4428) appear in roughly a quarter of all instructions. Other common spatial terms include "top" (3403), "bottom" (2058), "corner" (1190), "upper" (958), "middle" (827), and "edge" (714).
*   **Function and Brand:** Functional descriptions exist but are relatively rare (e.g., "washing machine" utilizing "washing" 100 times and "machine" 127 times; "scrubber" 28 times). Brand names are entirely absent from the vocabulary.

Real examples demonstrating this register include:
*   "put the blue knife in the orange pot on the lower right burner"
*   "move the silver pot between the knife and the red pepper"
*   "remove the cube from the white rectangular block and put it on top the blue block"

## Notable absences

Given that this is a corpus of instructions, several word classes and constructions are conspicuously missing:

*   **Politeness markers:** I looked for words like "please", "kindly", and "thanks", but none appear in the vocabulary table. This suggests the corpus conventions are strictly utilitarian and command-driven.
*   **Adverbs of manner and speed:** I looked for words dictating *how* an action should be performed, such as "carefully", "gently", "slowly", or "quickly", and found none. This suggests the corpus focuses entirely on the start and end states of objects rather than the kinematics of the movement itself.
*   **Conditionals and complex logic:** I looked for logical operators like "if", "unless", "until", and "while". Their absence indicates that the corpus consists of absolute, unconditional directives rather than reactive or state-dependent tasks.
*   **Exact measurements:** Despite the high frequency of spatial language, exact metric or imperial measurements are missing. I looked for "inches", "centimeters", "cm", and "degrees", but none are present. This suggests spatial relations are defined entirely relative to other objects and landmarks (e.g., "near", "beside", "between") rather than by absolute coordinates or distances.
*   **Brand names:** I looked for common household brands like "Coke", "Pepsi", or "Ikea", but found none, suggesting a reliance on generic physical attributes (color, shape, material) rather than commercial labels.
