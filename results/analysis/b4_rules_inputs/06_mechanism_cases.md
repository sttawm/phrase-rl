# Case studies: how wording changes the robot's success rate

All numbers below are measured success rates from running the robot policy in
simulation many times (hundreds of episodes per phrasing unless noted). Each
case compares different ways of phrasing the SAME task, always including the
task's own nominal instruction and its measured rate.

## Case 1: which word to use when two correct words exist
Nominal instruction: "stack the green block on the yellow block" → 24%.
The training data calls the objects "blocks", yet phrasings that say "cube"
("pick up the green cube and place it on top of the yellow cube" style)
succeed about 12 percentage points more than matched phrasings with "block",
and the best rewordings reach ~30-38%. Lesson: the policy has its own
preferred word for an object, which is not always the word its training data
— or the nominal instruction — uses. Both candidates here are short,
concrete nouns.

## Case 2: an object the training data has no word for (a ramekin)
Nominal instruction: "put coke can on ramekin" → 11% (n=864). The word
"ramekin" never appears in the training data, and the nominal fails badly.
- "Pick up the red cola can and place it upright inside the white bowl." → 64%
- Repeating a long input description unchanged ("...hollow white ceramic
  cup-like container...") → 42%
- "...place it in the white object" → 4%
Lesson: for an unfamiliar object, the best phrasing NAMES it with the
closest familiar concrete noun ("bowl") — beating the nominal instruction
itself by ~53 points. Repeating a rich description works moderately. A vague
category word like "object" is a disaster.

## Case 3: another unfamiliar object (a computer keyboard) — the opposite echo result
Nominal instruction: "put carrot on keyboard" → 18%. ("keyboard" also never
appears in the training data, but here the nominal is roughly the ceiling.)
- "...place it on the black keyboard." → 23%  (keep the unfamiliar word)
- "put the orange carrot on top of the keyboard" → 17%
- Repeating the input's description unchanged ("...black peripheral device
  used for typing") → 0.3%
- "...above the mouse pad" (a familiar-sounding but WRONG object name) → 3%
Lesson: unlike Case 2, no familiar substitute noun helps here — keeping the
unfamiliar object's actual name ("keyboard"), as the nominal does, is what
works. Describing the object instead fails completely, and substituting a
wrong familiar name is nearly as bad.

## Case 4: a third unfamiliar object (a wheel) — hard task, small numbers
Nominal instruction: "put carrot on wheel" → 16%.
- "pick the orange carrot and put it on the wheel" → 16% (ties nominal)
- "put it on the tire" style → 9%
- Rich description repeated → ~6%
All numbers are low (hard task). Naming the object plainly — as the nominal
does — is the ceiling; describing it or picking a different name loses.

## Case 5: small register differences matter (spoon on towel)
Nominal instruction: "put the spoon on the towel" → 56%.
- "put the spoon on top of the towel" → 64% (beats the nominal)
- "put the spoon onto the towel" → 60%
- "Place the spoon in the center of the towel." → 41%
- "place the spoon on the blue cloth" style → 25-34%
Lesson: short, plain, imperative phrasing wins, and a slight rewording can
even beat the nominal — but added precision words ("in the center of") and
object renamings ("cloth" for towel) cost real success.

## Case 6: length and complexity are enemies (can onto plate)
Nominal instruction: "put coke can on plate" → 65% (n=864).
Other phrasings tested for the same task (the can sits on the left, the
plate on the right — so the spatial details below are all accurate):
- "Retrieve the carbonated drink container on the left and position it
  carefully on top of the flat, yellow dining vessel to its right." → 1%
- "pick the can on the left side and put it on the yellow plate on the
  right side" → 10%
- "take the can on the left and put it on top of the yellow plate on the
  right" → 6%
Lesson: nothing recorded beats the short nominal. Long sentences lose
badly even when their extra spatial detail is accurate and potentially
useful — dropping the left/right qualifiers entirely beat keeping them.

## Case 7: cleaning up mild rewording is worth a lot (eggplant in basket)
Nominal instruction: "put eggplant into yellow basket" → 91%.
Other phrasings tested for the same task:
- "put the eggplant in the yellow basket" → 94% (slightly beats the nominal)
- "Arrange the eggplant neatly in the yellow bin." → 54%
Lesson: a mild style shift — a fancier verb ("arrange...neatly" for "put")
and a near-synonym for the receptacle ("bin" for "basket") — costs 40
points. Plain wording with the training data's own object names wins, and
the cleaned form even edged out the nominal.
