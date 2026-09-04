===RULES===

RULE 1 — OUTPUT FORMAT. ABSOLUTE.
Your entire reply is ONE line: the instruction to send to the robot. Nothing else.
No quotes. No label. No prefix. No bullet. No explanation. No reasoning. No thinking.
No alternatives. No second line. No blank line before or after.
If you decide nothing needs changing, you still reply with exactly one line: the
instruction itself. Never reply with an empty line. Start your reply with the first
word of the instruction and stop after its last word.

RULE 2 — SURFACE FORM. APPLIES TO EVERY OUTPUT, INCLUDING A PURE COPY.
The line you emit must be entirely lowercase.
It must end with no punctuation: no period, no question mark, no exclamation mark.
It must contain no comma, no semicolon, no colon, no quotation marks, no parentheses.
Collapse runs of spaces to one space.
Apply this even when you change nothing else.

RULE 3 — DEFECT LIST. RUN IT, THEN DECIDE.
Read the incoming instruction and check it against this list. Each defect names the
rule that repairs it.
  D1  a clause saying what NOT to do, what must stay put, or a condition
      (without, avoid, avoiding, keeping, while, rather than, provided, assuming,
      as long as, should you, if you can, do not, don't, so that, until,
      ensure that, make sure)                                            -> RULE 5
  D2  a manner, orientation, speed, path, sub-region, or source-location word
      or phrase                                                          -> RULE 6
  D3  a question mark, or politeness / first-person / passive-command framing
      (can you, could you, would you, will you, please, kindly, for me,
      i want you to, i need you to, your task is, it is required that,
      it is requested that, you are hereby instructed, proceed to, arrange for)
                                                                          -> RULE 7
  D4  two clauses joined by "and", or a lead-in verb before the placement verb
      (pick up X and ..., grab X and ..., take X and ..., lift X and ...,
      first ... then ...)                                                -> RULE 8
  D5  the opening verb is not "set"                                      -> RULE 9
  D6  an object is named by a rare, technical, formal or specialist word; by a
      description instead of a name; by a bare category word; or by a pronoun
      ("it", "one", "the green one", "the other one")                    -> RULE 10
  D7  a colour word that is compound or hedged ("light green", "teal-green",
      "off-white", "pale blue", "dark red"), or a material or size word
      (aluminum, tin, metal, ceramic, wooden, plastic, woven, small, large, little,
      toy, computer, dining)                                             -> RULE 12
  D8  no verb at all, or a missing "the" before either object              -> RULE 14
  D9  ornate, bureaucratic, indirect or long wording; more than one clause; a
      relative clause ("that", "which", "such that"); or three or more objects
      named                                                              -> RULE 15
Then:
  (a) NOTHING on the list fires -> do not rewrite. Apply RULE 2 and emit that line.
  (b) SOMETHING fires -> repair every defect that fired, using its rule, and change
      nothing else. Then run RULE 17 before you emit.
Do not invent defects that are not on this list. Do not "improve" wording that has
no listed defect.

RULE 4 — TARGET SHAPE. EVERY OUTPUT MUST END IN THIS SHAPE.
    <verb> the <colour?> <object-noun> <preposition> the <colour?> <destination-noun>
Exactly one verb. Exactly two noun phrases. Exactly one preposition. At most one
adjective on each noun, and that adjective may only be a colour word. No "and", no
comma, no second clause, no relative clause, no adverb.
Neutral illustrations of the SHAPE only (never copy these words unless the incoming
instruction is about these objects): "set the mug on the tray", "set the pen in the jar".
If your line does not fit this shape, keep deleting until it does.

RULE 5 — DELETE CONSTRAINT, NEGATION AND CONDITION CLAUSES.
Delete outright every span that says what not to do, what must not move, what must
stay clear, or under what condition to act. Delete the whole clause; never replace it
with a positive paraphrase; never mention the thing it protected. Keep only the
move-this-thing-to-that-thing content. This is the single most destructive feature in
the evidence: instructions carrying such a clause score near zero on every task.

RULE 6 — DELETE MANNER, ORIENTATION, PATH, SUB-REGION AND SOURCE WORDS.
Delete: upright, standing up, vertically, flat, sideways, level, the right way up,
gently, carefully, slowly, firmly, securely, safely, neatly, meticulously, directly,
swiftly, promptly, immediately, without delay, fully, completely, all the way, across
the workspace, across the table, over to, with your gripper.
Delete source locations: from the table, off the table, from its current location,
from its current position, from where it sits.
Replace sub-regions of the destination with the destination itself: "in the center of
the wheel" -> "on the wheel"; "on the surface of the plate" -> "on the plate"; "on the
edge of", "in the middle of", "on the left side of" -> plain contact with the whole thing.
Say only WHERE the object goes, never HOW it should sit or where it came from.

RULE 7 — EMIT A BARE COMMAND.
Delete every politeness word, question wrapper, first-person framing and passive
command listed in D3, and delete the question mark. The line must begin with the verb.

RULE 8 — COLLAPSE A PICK-UP LEAD-IN INTO ONE PLACEMENT.
If the instruction has the form
    <pick up | pick | grab | take | lift | grasp> the X and <put | place | set | lay |
    drop | leave | rest> it <PREPOSITION> the Y
delete the first clause and the word "it", and emit
    set the X <PREPOSITION> the Y
Do the same for longer chains ("pick up the X, carry it across the table, and place it
on the Y" -> "set the X on the Y").
Never emit two verbs. Never emit "and". Never mention grasping, lifting, carrying,
holding, the gripper, the arm, or the workspace: the policy needs the destination, not
the procedure. Never ADD a pick-up lead-in that was not there.

RULE 9 — THE VERB IS ALWAYS "set".
Whatever verb the input uses, emit "set". Replace put, place, lay, drop, rest, leave,
stick, slide, deposit, position, situate, arrange, stack, superimpose, move, transfer,
relocate, convey, carry, bring, transport, shift, hoist, elevate, maneuver -> "set".
Never emit "stack". Never emit "move X to Y". Never emit a second verb.
If the input already opens with "set", leave it.

RULE 10 — NAME EACH OBJECT WITH THE COMMONEST ORDINARY WORD FOR THAT SAME OBJECT.
One principle, one direction: replace a rare, technical, formal, regional or
descriptive name with the ordinary everyday word an ordinary person would use for that
same single object. Never move the other way, and never move up a level of generality.
Three hard prohibitions:
  10a. NEVER use a bare category word in place of a specific name. "the can", "the
       vegetable", "the block", "the cubes", "the utensil", "the container" name a
       class, not this object. Naming the class instead of the thing is one of the
       largest measured losses in the evidence.
  10b. NEVER use a pronoun or "one" for an object: "the yellow one", "the other one",
       "it". Write the full noun phrase both times, even though it repeats.
  10c. NEVER carry through a description in place of a name. Resolve it to the name:
       "elongated orange taproot" / "root vegetable" -> carrot;
       "bulbous purple vegetable" -> eggplant;
       "concave piece of silverware" / "the utensil" -> spoon;
       "device used for typing" -> keyboard;
       "red aluminum beverage cylinder" / "cylindrical cola container" /
       "red tin of cola" / "beverage container" -> coke can.
EXAMPLES of the permitted direction (these are instances of the principle, not the
whole rule — extrapolate the principle to words not listed):
       ramekin -> bowl;  bin -> basket;  dish / platter / saucer -> plate;
       block / brick / cuboid -> cube;  cola can / soda can / pop can / pepsi -> coke can;
       tire / rim -> wheel;  keys / keypad -> keyboard;  aubergine -> eggplant.
DEFAULT: if you are not sure the input's word is rare, technical or descriptive, then
it is ordinary — leave it exactly as it is. Do not swap one ordinary word for another
ordinary word (cup, bowl, cloth, towel and the like are all ordinary; leave whichever
one the input used). Never change WHICH object is moved or WHERE it goes.

RULE 11 — THE SCENE DESCRIPTION IS NOT A DICTIONARY.
Use the scene description for exactly two purposes:
  (i) to work out WHICH physical object a description refers to, so RULE 10 can name it;
  (ii) to read a destination colour under the strict test in RULE 12, and to tell two
       same-kind objects apart under RULE 12e.
Never take a noun from the scene description to replace a noun in the instruction.
Never copy positions, counts, materials, shapes, sizes, or any other object out of it.
Ignore entirely any list the description offers of alternative names, synonyms,
replacements, adjectives or verbs — those lists contain rare and losing words and are
not a naming authority. Read only its plain sentences describing the picture.

RULE 12 — COLOUR, MATERIAL AND SIZE.
  12a. KEEP every colour word already in the instruction. Never delete one. Removing a
       colour that was there is a measured loss.
  12b. NEVER add a colour to the moved object. Adding one is measured noise; skip it.
  12c. NEVER add a material or size word, and DELETE any that is present: aluminum,
       tin, metal, ceramic, wooden, plastic, woven, small, large, little, toy, computer,
       dining, dish (as in "dish rack").
  12d. A PLAIN COLOUR WORD is exactly one of: black, white, red, green, blue, yellow,
       orange, purple, brown, grey, silver, pink — standing alone as one word.
       "light green", "light yellow", "pale blue", "dark red", "off-white",
       "teal-green" and every hyphenated or two-word colour are NOT plain colour words.
       NEVER emit a colour that is not a plain colour word: if the instruction or the
       scene description gives a hedged or compound colour, drop the hedge only if one
       plain colour word remains unambiguous, otherwise drop the colour entirely.
  12e. ADD a colour to the DESTINATION only when BOTH of these hold:
         - the destination has no colour word already, AND
         - the scene description's plain sentences name that destination with ONE plain
           colour word (12d) and never attach any other colour, nor any hedged or
           compound colour, to that same destination.
       If the description hedges the colour, or uses two different colours for it, or
       you are unsure, leave the destination bare. Adding a stable plain colour to a
       destination is worth about ten points; adding an unstable or hedged one costs
       about twenty. When in doubt, bare.
  12f. When the moved object and the destination are the SAME KIND of thing (a cube on
       a cube, a can on a can), both must be told apart. Keep both colours the input
       gives. If the input gives none, take one plain colour word (12d) for each from
       the scene description's plain sentences. An instruction that does not
       distinguish two same-kind objects fails almost completely.

RULE 13 — PREPOSITION.
Keep the preposition the input already uses; do not swap on / onto / on top of / atop,
and do not swap in / inside / into. The differences are measured noise.
Choose one only when the input gives none, or when a rule above deleted it. Then use
this test:
  - if the destination is a thing the object goes INSIDE (bowl, basket, cup, box, bin,
    jar, pot, tub — anything with walls that hold the object), use "in";
  - if the destination is a surface the object RESTS ON TOP OF (plate, towel, keyboard,
    wheel, table, another cube — anything flat or open), use "on".
Apply the test by asking whether the object ends up inside the thing or on top of it —
do not rely on a list of names. Never write at, near, toward, over, against, or across.

RULE 14 — KEEP THE VERB AND THE ARTICLES.
The line must begin with a verb and must contain "the" before the moved object and
"the" before the destination. If the input has no verb ("carrot on keyboard", "green
cube on yellow cube"), add "set" and add the missing "the"s. Never emit a telegraphic
or headline-style string; dropping the verb is a large measured loss.
Replace "a", "that", "this", "your" before an object with "the".

RULE 15 — REBUILD ORNATE, INDIRECT OR LONG INSTRUCTIONS FROM SCRATCH.
When D9 fires, do not edit the input word by word. Work out the two things — what is
moved, and where it must end up — and emit exactly the RULE 4 shape:
    set the <colour?> <object> <on|in> the <colour?> <destination>
carrying only colours that survive RULE 12, and choosing the preposition by RULE 13.
Drop everything else in the input, including any lead-in, any third object, and any
framing. This is the largest repair in the evidence: this class of instruction scores
near zero as written and roughly a third of episodes succeed after rebuilding.

RULE 16 — EXACTLY TWO OBJECTS.
Name the moved object once and the destination once. Never name a third object. Never
offer alternatives ("the plate or the dish") — if the input names two candidate
destinations, keep the first and delete the rest. Never mention the table, the
workspace, the gripper, the arm, or any object the instruction did not mention.

RULE 17 — FINAL CHECK, THEN TIE-BREAK.
Before emitting, confirm ALL of these about your line:
  starts with "set" (or with the input's verb if you copied under RULE 3a);
  one clause; no "and"; no comma; no "without"; no "?"; no adverb;
  all lowercase; no punctuation at the end;
  exactly two noun phrases, each preceded by "the";
  at most one adjective per noun, and every adjective is a plain colour word;
  the same object and the same destination as the input.
If any check fails, fix it and check again.
TIE-BREAK: if two rules point different ways, if you cannot tell whether a defect
fires, or if you cannot tell which object is moved and which is the destination, emit
the input with RULE 2 applied and nothing else. A wrong edit has a measured cost; a
skipped edit only forgoes a gain.
Now emit that one line and nothing else (RULE 1).

===RATIONALE===

Evidence: /Users/sttawm/dev/robotics/phrase-rl/results/rules_runs/v2distill/sim_only/evidence.csv,
516 phrases over 8 WidowX tasks, every row a real rollout success rate, n_ctx 6-438,
28,656 scored contexts total. All figures below are my own recomputation, using
same-task same-frame matched pairs aggregated by normalised phrase and weighted by the
smaller side's n_ctx. Success values are multiples of 2.78 (36-episode batches), so a
single pair carries roughly +/-8 points; nothing below rests on one pair.

R2 surface. Matched pairs differing only in case/terminal period, both sides n>=30:
6/6 favour lowercase-no-period, mean +9.7. Largest: "Place the carrot on the keyboard."
0.0 (n=96) vs lowercase 13.9 (n=78); "put the coke can on the plate." 47.2 (n=64) vs
61.1 (n=304); "place the red cola can inside the white bowl." 41.7 (n=64) vs 61.1 (n=70).

R3 gate. The "never touch a fluent input" framing that two candidates built on is
confounded and I rejected it. Pooled, rewriting natural bases GAINS: 29.9 (n=3580) ->
33.0 (n=624), improving in 5 of 7 non-saturated tasks (keyboard 0.0->16.7, wheel
0.0->25.9, carrot-on-plate 16.7->30.0, spoon 41.4->44.4, eggplant 91.7->100.0). The
single counterexample (coke-on-plate 66.7 -> 24.1) is one task, and its losing rewrites
say "dish", "soda", "light green plate", or carry a capital and a period — every one an
edit this book already forbids. So the gate is a surface defect list, not a judgement
about fluency, which the weakest applier cannot make.

R5 constraints. Rows containing a constraint/negation/condition clause: 19 rows,
n_ctx 2648, weighted 6.6 success, 33.2 points below their own task medians. The most
destructive feature measured, by a wide margin.

R6 manner/path. Manner and orientation words present: 32 rows, n_ctx 2866, 21.3 below
task median. Inserting "upright": 0 of 4 pairs positive, mean -5.7. Sub-regions:
"in the center of the wheel" 0.0 vs "on the wheel" 16.7; "on the surface of the plate"
16.7 vs "on the plate" 38.9. Source locations: "take the orange carrot from its current
location and put it on the wheel" 0.0.

R7 framing. Question/politeness/first-person rows: 6 rows, n_ctx 994, 8.2 below task
median. "can you put the carrot on the wheel for me" 0.0 (n=78) vs "put the carrot on
the wheel" 16.7 (n=348). Note "please" alone is neutral; the wrapper is the damage.

R8 pick-up collapse. 42 exact-frame matched pairs across 7 tasks, mean +8.7,
n_ctx-weighted +9.2, 31/42 positive. Deepest: "take the coke can and put it on the
plate" 30.6 (n=64) -> "put the coke can on the plate" 58.3 (n=438); "pick up the spoon
and put it on the towel" 16.7 (n=82) -> 44.1 (n=118); "pick up the carrot and set it
down on the keyboard" 0.0 (n=246) -> "set the carrot on the keyboard" 13.9 (n=264).
Feature level: lead-in present, 96 rows, n_ctx 3648, 6.0 below task median. One
candidate forbade this repair on two pairs whose stripped sides have n=18 and n=6.

R9 verb. Matched leading-verb substitutions: put->set +4.8 (26 pairs, 8 tasks, 18/26),
place->set +6.6 (23 pairs, 8 tasks), lay->set +6.9 (10 pairs), drop->set +9.5 (9 pairs,
8/9), rest->set +6.5 (3 pairs), stack->place +21.8 (6 pairs, 6/6). Transport verbs
present: 27 rows, n_ctx 2186, 13.8 below task median. "stack the cubes" 0.0 (n=44).
"set" is the only verb positive against every alternative tested, on 8 of 8 tasks.

R10 nouns. One direction, toward the commoner ordinary word for the same object:
ramekin->bowl +33.0 (3 pairs), bin->basket +38.9 (1 pair), dish->plate +24.1 (4 pairs,
2 tasks, 4/4), block->cube +12.8 (5 pairs, 5/5), cola->coke +15.6 (8 pairs, 2 tasks),
soda->coke +11.4 (4 pairs), pepsi->coke +8.3, tire->wheel +6.7 (5 pairs),
keys->keyboard +6.4. Dropped as unsupported: cup->bowl +2.3 (2 pairs, 1/2) and
cloth->towel +0.9 (6 pairs, 2/6) — hence "do not swap one ordinary word for another".
Category words and pronouns: "put the can on the plate" 30.6 (n=64) vs "put the coke
can on the plate" 61.1 (n=304); "move the orange vegetable to the wheel" 0.0 (n=252);
"put the green one on top of the yellow one" 33.3 (n=230) vs "put the green cube on the
yellow cube" 38.9 (n=68); "stack the cubes" 0.0 (n=44). Circumlocution rows: 11 rows,
n_ctx 1490, weighted 21.5, and 0.0 at n=204-208 on four tasks.

R11 trace discipline. I read the file the applier is actually fed
(results/phrase_artifacts/traces_rules_v1.parquet, 2641 rows). Every trace ends with an
explicit "Nouns / Verbs / Adjectives: potential replacements" list, and those lists
supply the losing words: of 53 ramekin-task traces, 52 contain "bowl" but 40 contain
"ramekin", 39 "cup", 22 "basin"; a plate-task trace reads "light yellow circular plate"
and then offers "yellow plate, light green dish, saucer, circular tray". A rule of the
form "use the trace's noun" is therefore both inexecutable and pointed at the losers.

R12 colour. Keeping is what matters, not adding. Removal is a real loss: -black wheel
-15.0 (7 pairs, 1/7 positive), -black keyboard -7.7 (9 pairs, 1/9), -yellow basket
-16.6, -"red" from "red cola can" -14.8 (5 pairs, 1/5). Adding to the moved object is
noise: "the carrot" -> "the orange carrot" +0.4 mean over 24 pairs, 12/24 positive.
Adding to the destination splits by task: black wheel +15.0 (6/7), black keyboard +7.7
(7/9), yellow basket +16.6, white bowl +3.9, blue towel +4.0 — against yellow plate
-15.8 (5 pairs, 1/5) and light-green plate -23.1 (3 pairs, 0/3). The separator is
observable in one trace: counting colour words in the description sentences of each
task's traces, wheel is "black" (16 traces, no competing colour), keyboard "black",
basket "yellow", bowl "white" — while BOTH plate tasks produce only hedged colours
("light green" 7, "light yellow" 3, zero unhedged hits) and the plate traces disagree
with each other. Hence 12d/12e: one plain unhedged colour word or nothing. Hedged
colours also lose on the object: "put the teal-green cube on the yellow cube" 22.2 vs
"put the green cube on the yellow cube" 38.9. Material/size words are dead weight
("small teal-green cube" 19.4). Same-kind disambiguation (12f) is mandatory:
"cube on cube" 8.3 (n=44), "stack the cubes" 0.0 (n=44) vs coloured variants 25.0-58.3.

R13 prepositions. All noise, so copy: on->onto +4.1 (13 pairs, 8/13), on->on top of
+1.8 (33 pairs, 17/33), on->atop +6.9 (4 pairs), in->inside -5.7 (6 pairs, 2/6),
in->into +1.0 (3 pairs). Choosing wrongly is not noise: "put coke can on ramekin" 0.0.
The rule is stated as a geometric test, not a list of container names, so a novel
receptacle does not fall through to "on".

R14 verb and articles. "eggplant in yellow basket" 61.1 (n=64) vs "put the eggplant in
the yellow basket" 97.2 (n=64) — the deepest matched contrast in the corpus. Dropping
articles alone is near-neutral ("put carrot on plate" 41.7 vs "put the carrot on the
plate" 38.9), so the rule restores them without fuss and never removes them to shorten.

R15 rebuild. Adversarial bases: 125 rows, n_ctx 4498, weighted 17.1, and exactly 0.0 at
n=204-752 on keyboard, wheel, coke-on-plate, coke-on-ramekin and stack-cube. Their
rewrites: 33.9 (n_ctx 852). Per task 0.0->26.7 keyboard, 0.0->16.7 wheel, 0.0->37.1
coke-on-plate, 0.0->22.9 ramekin, 0.0->23.7 stack, 12.5->42.2 spoon, 20.8->26.8
carrot-on-plate. This is the book's whole upside.

R4/R17 shape instead of a word count. Length relative to task median: <=4 words -3.6,
5 +2.9, 6-7 +1.4, 8-9 +3.9, 10-11 -3.0, 12-14 -2.0, 15-17 -23.1, 18+ -32.6 (16 rows,
n_ctx 3052). The RULE 4 shape lands in the 6-9 band without asking a greedy 9B with
thinking disabled and a 64-token budget to count words.

R16 two objects. "put the carrot on the plate or the dish" 25.0 vs "put the carrot on
the plate" 38.9; "put the coke can on the plate or the dish" 41.7 (n=36) vs 61.1
(n=304); "put the spoon on the towel or the cloth" 58.3 vs 75.0.

LEAKAGE CONTROL. Only the ===RULES=== body reaches the applier — scripts/rules_loop_jobs.py
splits on the markers and drops this section, with the comment that feeding the applier
commentary is "a live hazard". The rules body therefore contains no measured phrase, no
score, and no complete instruction drawn from any of the 8 evaluation tasks. The two
shape illustrations in RULE 4 use objects that appear nowhere in the corpus ("mug on
tray", "pen in jar") precisely so a greedy applier cannot pattern-match a memorised
winning line. Word-level noun arrows in RULE 10 are single words, labelled as examples
of a stated principle.

WHERE THIS BOOK IS A HYPOTHESIS, NOT A LAW.
- widowx_put_eggplant_in_basket is saturated: its adversarial base scores 100.0 at
  n=208 and every phrasing lands 91-100. No rule is validated there; the only things
  drawn from it are matched-pair contrasts (verbless form, basket colour).
- ramekin->bowl (3 pairs, 1 task), block->cube (5 pairs, 1 task), tire->wheel (5 pairs,
  1 task), bin->basket (1 pair) and the destination-colour effect each rest on one or
  two tasks. The transferable claim is the principle in RULE 10 and the surface test in
  RULE 12d/e; the arrows are its instances.
- 157 of 516 rows sit at n_ctx=6. No claim above rests on an n=6 contrast alone.
- The RULE 12e trace test is validated on 5 destinations across 7 tasks. It is the rule
  most likely to backfire on a novel object, which is why its default is "leave bare".
- Not ruled on, because the measurements are split: on vs onto vs on top of vs atop,
  in vs inside vs into, inserting "down", dropping articles, cloth vs towel, cup vs
  bowl. Forcing any of these would add churn with no measured gain.