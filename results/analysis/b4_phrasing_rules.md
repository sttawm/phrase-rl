# B4 phrasing rules for the instruction rewriter

Scope: these rules convert an adversarially reworded instruction, plus its trace
(scene sentence + description→plain-name mapping, format as in
`07_selftraces_val8.md`), into the phrasing the frozen policy executes best.
Every rule is executable from only the incoming instruction, the trace, and the
training corpus (`01_bridge_gt_instructions.txt`, `02_oxe_paraphrases_SAMPLE60keys.csv`).
No rule requires rollout measurements.

**Application order.** (1) Use the trace mapping to resolve every object
reference to a plain name (Rules 1–3). (2) Choose the verb frame (Rule 4) and
the placement relation (Rules 6–7). (3) Strip everything the task does not need
(Rules 5, 8). (4) Add at most minimal modifiers (Rule 9). (5) Emit one short
bare imperative (Rule 10). When two rules pull in opposite directions, see
"Known tensions".

---

## Rule 1 — Name, never describe: replace every object description with the trace's plain name

**Rule.** Every multi-word description of an object ("the hollow cylindrical
container used for drinking") must be replaced by the plain concrete noun the
trace maps it to. No description, however accurate, may survive into the
output. The trace's name is the default even when the incoming instruction's
own noun differs, because the trace names what the object visually is.

**When it applies.** Always; this is the rewriter's core move.

**Worked example.** Incoming: "Locate the yellow curved fruit and set it down
on the woven rectangular surface." Trace maps "yellow curved fruit" → banana,
"woven rectangular surface" → placemat. Output: "put the banana on the
placemat."

**Evidence.** 06 Cases 2–4: echoing a rich description scores 42% at best
(Case 2) and 0.3–6% typically (Cases 3–4) versus 23–64% for the plain name.
03: named phrasings beat description phrasings with the largest deltas in the
whole file (delta 63.5pp, z=17.7; delta 22.6pp, z=10.0). 04: every
description-style row is at the bottom of its task block (1.0%, 0.3%). 07:
every trace supplies exactly the needed mapping.

## Rule 2 — Never move up the abstraction ladder

**Rule.** Never use a category noun — "object", "thing", "item", "element",
"container", "device", or a class word like "vegetable"/"fruit"/"utensil" — in
place of a specific name. If the trace's name is itself an unusual word absent
from the corpus, still prefer that unusual concrete name over any familiar
vague one.

**When it applies.** Whenever tempted to hedge because the object seems
unfamiliar or the mapping seems uncertain.

**Worked example.** Trace maps "the gripping tool with rubber jaws" → pliers
("pliers" is corpus-absent). Output says "the pliers" — not "the tool", not
"the metal object". "put the pliers on the tray", not "put the gray thing on
the tray."

**Evidence.** 06 Case 2: "...place it in the white object" → 4% versus 64% for
a concrete noun. 06 Case 3: keeping corpus-absent "keyboard" (23%) crushes
describing it (0.3%). 03: "Place the orange carrot on the green plate."
(41.7%) beats "Place the vegetable on the ceramic dish." (16.7%, z=3.73). 05:
"orange vegetable" variants score 4–12% where "orange carrot" variants score
45%+. 01: "object" (1,527 lines) and "thing" (320) do appear in the corpus,
yet still fail measured — corpus frequency does not rescue vagueness, so this
rule outranks frequency arguments.

## Rule 3 — Only objects from the trace; never a lookalike substitute

**Rule.** The output may mention only objects present in the trace's mapping
or scene sentence, using their mapped names. Never swap in a different
familiar object's name because it sounds plausible for the scene, and never
invent metaphors.

**When it applies.** Always; check the final output against the trace's object
list before emitting.

**Worked example.** Trace maps "the writing surface" → whiteboard. Do not
output "put the marker on the desk" (desk is not in the trace) or "put the
marker on the notebook" (wrong object). Output: "put the marker on the
whiteboard."

**Evidence.** 06 Case 3: substituting the familiar-sounding but wrong "mouse
pad" → 3% versus 23% for the right name. 05 (structure only): hallucinated
renamings score 0–8% across tasks — "mouse", "bell pepper", "banana",
"radish", "garlic", "laptop", "traffic cone", "leaf"/"sun" metaphor rows all
sit at the bottom of their pools.

## Rule 4 — Verb frame: a high-frequency corpus verb in a bare imperative

**Rule.** Build the sentence on one of the corpus-dominant frames:
"put X on/in Y", "place X on/in Y", "move X to Y", "take X and put it on Y",
or "pick up X and place/put it on/in Y". If the incoming instruction already
uses a plain physical verb that fits the action and is corpus-attested (e.g.
"stack"), keeping it is fine. Never use search-, ceremony-, or
precision-verbs: retrieve, locate, grasp, arrange, position, set, balance,
deposit, secure, maneuver, relocate, transport. Never add politeness
("please"), questions, meta-commentary, or any text that is not the
instruction itself.

**When it applies.** Every output sentence.

**Worked example.** Incoming: "Kindly retrieve the sponge and position it atop
the saucepan lid." Output: "pick up the sponge and place it on the pot lid."

**Evidence.** 01 first-word counts: move 6,626 / put 3,393 / place 1,326 /
take 1,039 / pick 874, versus retrieve 0, locate 0, grasp 0, arrange 4,
balance 1, deposit 0, position 24, set 17; "please" and "you(r)" ~0. 03/04:
"Retrieve..." 1.0–1.1%, "Locate..." 0.0–0.3%, "Arrange..." loses to a
bin-matched "Pick up...place" control by 19–29pp (z=4.09, z=5.77); "Lift the
red can..." loses to "Pick up the red cola can..." by 43pp (z=6.71). 02: the
augmentation corpus is full of these verbs (position 402, shift 200, transfer
171, arrange 110, carefully 57) — the policy has seen them, yet they still
fail; only file-01-style phrasing predicts success, so calibrate on 01, not
02. 05: meta/format rows ("This is a reworded robot instruction...", a bare
"(spoon, towel)" tuple, "Please place...") score 0–25%; non-imperative forms
(question, "The X should be...") underperform their imperative twins.

## Rule 5 — Strip manner adverbs and ceremony words

**Rule.** Delete adverbs and adverbial padding describing how to move:
"carefully", "gently", "neatly", "precisely", "exactly", "securely",
"slowly", plus filler like "kindly", "make sure to", "so that it is". They
never help and often collapse success.

**When it applies.** Whenever the incoming instruction contains any manner or
emphasis wording.

**Worked example.** Incoming: "Gently and precisely nestle the mug onto the
coaster." Output: "put the mug on the coaster."

**Evidence.** 01: "carefully", "neatly", "gently", "exactly", "precisely" all
occur 0 times in 17,297 instructions — they are out-of-distribution tokens.
04: "Arrange the eggplant neatly in the yellow bin." 53.8% versus 62.2% for
the same sentence without "neatly" (03, z-cleared against controls); 05:
"Arrange the eggplant in yellow bin with neatness." 8.3%. 03: "Set the spoon
exactly in the middle of the towel." loses to "put the spoon on the towel" by
31.7pp (z=3.48). 06 Cases 5–7 all show the plain form winning.

## Rule 6 — Strip placement-precision goals; keep the coarse relation

**Rule.** Reduce the goal to a coarse contact relation: on / on top of / onto
/ in / into / inside. Delete precision targets — "in the center of", "in the
middle of", "exactly on", distances, alignment demands — even when they are
accurate and appear verbatim in the incoming instruction. Also never use
non-contact prepositions ("above", "over", "near") for a placing action.

**When it applies.** Any placement task whose incoming form demands a precise
final position.

**Worked example.** Incoming: "Deposit the hairbrush exactly in the middle of
the tray." Output: "put the hairbrush on the tray."

**Evidence.** 03: "in the center of the tire" phrasings measure 0.1% (n=936)
and 3.3% (n=1,224) against 16–29% for on-the-tire forms (z up to 4.56);
"place the spoon on the towel" beats "Pick up the spoon and place it in the
middle of the towel." by 40.6pp (z=4.79). 04: every "center/middle" row sits
below its task's plain-relation rows. 01: "in the center" appears 21 times
and "in the middle" 175 times in 17,297 lines — rare; "on top of" appears
1,285 times — safe. 04: "put it above the keyboard" 1.4% and "above the mouse
pad" 3.1% versus "on the black keyboard" 25.3%; 01 has "above" only 329
times, mostly for relative positions rather than placement goals.

## Rule 7 — Containment tasks: "inside", plus final-pose words when the input implies orientation

**Rule.** When the goal is putting an object into a container (trace name is a
bowl-, cup-, jar-, bin-, or basket-like receptacle), use "place it inside the
[container]" / "put it in the [container]". If the incoming instruction
implies a final orientation (standing, upright, vertical), keep exactly one
orientation word for the moved object — "upright" or "vertically". Orientation
words describe the end state of the object and help; do not confuse them with
the position-precision words banned by Rule 6.

**When it applies.** Container placements only; on flat surfaces, plain "on"
(Rule 6) applies.

**Worked example.** Incoming: "Transfer the marker so it stands erect within
the glass jar." Trace: "glass jar" → jar. Output: "pick up the marker and
place it upright inside the jar."

**Evidence.** 04 (ramekin task): "...place it upright inside the white bowl."
64.2% and "...vertically inside..." 63.5% versus "...place it inside..."
46.2% and "...on the white bowl" 31.9% (03: upright-inside beats on-the-bowl,
z=4.34). 05 (structure): all top ramekin draws use upright/standing/vertically
+ inside (58–79%). 01: "inside" appears 413 times; "upright" only 10 and
"vertically" 3 — one of the few places measured evidence overrides corpus
frequency, so cap it at a single orientation word.

## Rule 8 — Drop current-location and route qualifiers

**Rule.** Delete phrases saying where objects currently are or how to travel
— "on the left", "to its right", "from the sink", "nearby", "over to" —
unless two objects in the trace would otherwise be indistinguishable. The
policy sees the scene; wording only needs to identify objects and the goal
relation.

**When it applies.** Whenever the incoming instruction embeds spatial
bookkeeping; nearly all verbose adversarial inputs do.

**Worked example.** Incoming: "Take the mug sitting on the left edge and
carry it across to the saucer on the far right." Output: "put the mug on the
saucer."

**Evidence.** 06 Case 6: accurate left/right qualifiers cost 55–60pp
("...can on the left side...plate on the right side" 10.1% versus 64.6%
without them). 03: "Pick up the red cola can and place it on the yellow
plate." beats the same sentence with "on the left...to its right" appended by
10pp (z=2.11) — an otherwise-identical minimal pair. 01: "left" (4,637) and
"right" (4,428) are corpus-frequent, yet the qualifiers still hurt — the cost
is added clause weight, not word rarity, which is why deletion (not synonym
choice) is the fix.

## Rule 9 — Modifiers: one color adjective where identity needs it, bare "the" elsewhere

**Rule.** Attach exactly one color/appearance adjective (taken from the trace
scene sentence) to an object when (a) its name is rare or absent in the
corpus, or (b) the scene contains a same-shape sibling distinguished only by
appearance — in the sibling case both objects keep their color words. For
corpus-common, unambiguous objects use the bare noun with "the". Never stack
more than one adjective on a noun.

**When it applies.** After naming (Rules 1–3), as the only permitted addition.

**Worked example.** Trace: scene has a gray stapler ("stapler" corpus-absent)
and one folder. Output: "put the gray stapler on the folder" — color on the
rare noun, none on the common one. If the scene had two folders, one red and
one blue: "put the gray stapler on the red folder."

**Evidence.** 03 minimal pairs: "black keyboard" beats bare "keyboard"
(+10.5pp, z=2.36); "center of the black tire" beats "center of the tire"
(+12.4pp, z=4.33) — color rescues corpus-absent nouns. 04: winning
ramekin-substitute is "white bowl"; top wheel row is "the black tire" 29.2%.
For the same-shape sibling case, every top stack-task row in 04/05 keeps both
color words, and 07's stack trace supplies "teal-green"/"yellow". Conversely
03: the adjective-free "put coke can on plate" beats "Pick up the red cola
can and place it on the yellow plate." by 27.4pp (z=5.41), and plain "put
eggplant into yellow basket" beats "Pick up the purple eggplant..." (+9.0pp,
z=3.32) — for familiar objects, added adjectives are dead weight.

## Rule 10 — One short sentence in corpus register; minimal edit from a training-style utterance

**Rule.** Emit exactly one imperative sentence, target 4–12 words (never
beyond ~14), at most two clauses (the second only the corpus frame "and
place/put it ..."), no relative clauses, no "so that", no second sentence.
Prefer the shortest output consistent with Rules 1–9; when the incoming
instruction already reduces to a short plain corpus-style clause with
trace-consistent names, keep that reduction unchanged rather than rephrasing
further. Single-clause beats two-clause when both are natural.

**When it applies.** Final emission check for every rewrite.

**Worked example.** Incoming: "I'd like you to first identify the sponge and
then, once located, ensure it ends up resting on the skillet that is nearest
to you." Output: "put the sponge on the pan." (5 words, one clause.)

**Evidence.** 01: median instruction length is 10 words, 82.9% are ≤12 words,
and only 93 of 17,297 lines contain two "and"s. 06 Case 6 and 03's largest
deltas: 18–20-word sentences lose by 22–63pp to short forms regardless of
content. 03 single- vs two-clause minimal pairs: "Place the green block on
top of the yellow block." beats "Pick up the green block and place it on top
of the yellow block." (+11.6pp, z=2.12). 06 Cases 5 and 7: a minimal cleanup
("put the spoon on top of the towel" 64%, "put the eggplant in the yellow
basket" 94%) can even beat the task's own nominal — mild normalization is
worth doing, aggressive rephrasing is not.

---

# Known tensions and arbitration

**T1. Substitute a familiar name vs keep the unfamiliar one (06 Case 2 vs
Cases 3–4).** For the ramekin, renaming to "white bowl" gained ~53pp; for the
keyboard and wheel, keeping "keyboard"/"tire" was the ceiling and familiar
substitutes ("mouse pad") failed. The rewriter cannot see success rates, so
arbitrate through the trace: use the name the trace mapping assigns (Rule 1).
The trace called the ramekin a "ceramic bowl" and the keyboard a "keyboard"
(07) — its vision-grounded name already encodes whether a familiar noun
visually fits. If the incoming instruction's own head noun is a plain
concrete noun that agrees with the trace, keep it; substitute only when the
incoming form is a description or the trace maps it elsewhere. Never resolve
this tension by retreating to a category word (Rule 2 is absolute).

**T2. Terse telegram vs fully-specified pick-and-place.** "put coke can on
plate" (64.6%) crushed every longer, adjective-rich variant (03, z up to
8.3), yet the ramekin winner was a 13-word colored two-clause sentence.
Arbitration: default to the shortest form (Rule 10) with bare nouns (Rule 9);
spend words only where Rules 7 and 9 license them — containment orientation
and rare/ambiguous-noun colors. Length is a budget: every added word must buy
object identity or goal relation.

**T3. Corpus frequency is a guide, not a law.** Frequency correctly bans
retrieve/locate/arrange/"neatly" (all ~0 in 01, all measured disasters) and
prefers "basket" (305) over "bin" (4) — bin-phrasings lose CI-cleared pairs
in 03. But "upright" (10 occurrences) helps containment (Rule 7), "stack" (9)
performed fine when it was the incoming verb, corpus-frequent "object"
(1,527 lines) still fails (Rule 2), and one 04 cell with corpus-absent
"aubergine"/"dish rack" scored 92.7%. Arbitration: frequency decides between
otherwise-equal synonyms (concrete noun vs concrete noun, verb vs verb);
it never overrides the semantic rules (name > describe, concrete > abstract,
orientation words for containment) nor licenses adding words.

**T4. Orientation precision vs position precision (Rule 7 vs Rule 6).**
"upright inside the bowl" helps; "in the center of the tire" is fatal. The
line: words describing the moved object's final pose (upright, vertically)
may stay — one at most; words constraining where on the target it must land
(center, middle, exactly, edge offsets) always go. When unsure which kind a
word is, delete it: 05's pools show deletion-style drafts clustering at the
top and precision drafts at 0%, and no measured case punishes dropping a
qualifier the way "center" rows punish keeping one.

**T5. When two concrete nouns both fit (06 Case 1).** The corpus's own word
("block", 868) was measurably a shade worse than "cube" (713) — the policy's
preferred word is not always the corpus's. Both words are short, concrete,
and corpus-frequent, and the measured gap is small (within the noise band of
04). Arbitration: prefer the noun the trace's scene sentence itself uses
(07's trace says "two small cubes"), since it reflects what the object looks
like; frequency breaks any remaining tie. Expected cost of a wrong choice
here is a few points, unlike T1–T4 where wrong choices cost tens of points.
