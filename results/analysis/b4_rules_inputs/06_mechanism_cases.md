# Case studies: how wording changes the robot's success rate

All numbers below are measured success rates from running the robot policy in
simulation many times (hundreds of episodes per phrasing unless noted). Each
case compares different ways of phrasing the SAME task.

## Case 1: which word to use when two correct words exist
Task: stack the green block on the yellow block.
The training data calls the objects "blocks". But phrasings that say "cube"
succeed about 12 percentage points MORE than phrasings that say "block" —
even though "block" is what the training data says. Lesson: the policy has
its own preferred word for an object, which is not always the word its
training data used most. Both candidates here are short, concrete nouns.

## Case 2: an object the training data has no word for (a ramekin)
Task: put a soda can into a small white ceramic dish (a "ramekin" — a word
that never appears anywhere in the training data).
- "Pick up the red cola can and place it upright inside the white bowl." → 64%
- Repeating the long input description unchanged ("...hollow white ceramic
  cup-like container...") → 42%
- "...place it in the white object" → 4%
Lesson: for an unfamiliar object, the best move is to NAME it with the
closest familiar concrete noun ("bowl"). Repeating a rich description works
moderately. A vague category word like "object" is a disaster.

## Case 3: another unfamiliar object (a computer keyboard) — the opposite echo result
Task: put a carrot on a computer keyboard ("keyboard" also never appears in
the training data).
- "...place it on the black keyboard." → 23%  (keep the unfamiliar word)
- "put the orange carrot on top of the keyboard" → 17%
- Repeating the input's description unchanged ("...black peripheral device
  used for typing") → 0.3%
- "...above the mouse pad" (a familiar-sounding but WRONG object name) → 3%
Lesson: unlike Case 2, repeating the description fails completely here — the
policy understands some descriptions and not others. Keeping the unfamiliar
object's actual name ("keyboard") works best. Substituting a wrong familiar
name is nearly as bad as the description.

## Case 4: a third unfamiliar object (a wheel/tire) — hard task, small numbers
Task: put a carrot on a wheel.
- "pick the orange carrot and put it on the wheel" → 16% (best recorded)
- "put it on the tire" style → 9%
- Rich description repeated → ~6%
All numbers are low (hard task), but naming the object beats describing it.

## Case 5: small register differences matter (spoon on towel)
Task: put the spoon on the towel.
- "put the spoon on top of the towel" → 64%
- "put the spoon onto the towel" → 60%
- "Place the spoon in the center of the towel." → 41%
- "place the spoon on the blue cloth" style → 25-34%
Lesson: short, plain, imperative phrasing in the training data's style wins;
added precision words ("in the center of") and object renamings ("cloth" for
towel) cost real success.

## Case 6: length and complexity are enemies (can onto plate)
Task: put a coke can on a plate. The input arrived as a long, convoluted
sentence with directions ("the container on the left ... vessel to its
right").
- "put coke can on plate" → 65% (n=864)
- Long phrasings that carefully preserve the left/right directions → 6-10%
- Repeating the convoluted input unchanged → 1%
Lesson: even when a long phrasing is accurate and preserves useful spatial
detail, it loses badly to a short canonical form. Dropping the spatial
qualifiers entirely was better than keeping them.

## Case 7: cleaning up mild rewording is worth a lot (eggplant in basket)
Task: put an eggplant into a yellow basket. Input said "Arrange the eggplant
neatly in the yellow bin."
- "put the eggplant in the yellow basket" → 94%
- Repeating the mildly-reworded input unchanged → 54%
Lesson: converting even a MILD rewording back to the training data's plain
style ("arrange...neatly"→"put", "bin"→"basket") gained 40 points.
