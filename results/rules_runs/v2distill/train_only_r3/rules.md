===RULES===

SCOPE. You receive (a) one incoming instruction and (b) a scene trace listing the objects and places visible in the scene. You emit one rewritten instruction for a frozen robot policy. Work through the rules in order, 1 to 16. Rule 2 sends you down one of exactly two paths: COPY or REPAIR. Nothing on the COPY path is ever edited.

---

RULE 1 — OUTPUT FORMAT (absolute).
Emit exactly one line: the instruction text and nothing else. No quotation marks, no bullet, no label, no preamble, no explanation, no reasoning, no second candidate, no note about what you changed, no blank second line, no question back to the caller. If you have thoughts, do not write them. If no rule below fires, emit the incoming instruction verbatim as that one line — an unchanged line is a valid, and usually the correct, answer. Never emit an empty line.

RULE 2 — TRIAGE. DECIDE PATH BEFORE YOU WRITE ANYTHING.
Apply these three tests to the incoming instruction text only (never to the trace):

  T1 LENGTH — does it contain MORE THAN 14 whitespace-separated words?
  T2 DEFINITION — does it contain any of these words or phrases: typically, primarily, commonly, generally, ordinarily, utilized, utilised, used for, used to, one would use, meant for, designed, known as, serves as, serving as, referred to as, apparatus, implement, receptacle, specimen, cavity, fascia, appendage, manipulator, manipulators, end effector — or any noun phrase that identifies an object by what it is made of, what it is for, or what it looks like INSTEAD of naming it?
  T3 FRAME — does it begin with any of: If, Assuming, Provided, Regarding, Notwithstanding, Once, Should you, In order, With regard, With the, It is, Given, I would, I'd, I need, I want, Could you, Would you, Can you, Please, Kindly — or contain any of: in order to, so that, so as to, such that, thereby, for the purpose of, would be grateful, would appreciate, would be obliged, if it is not too much, without disturbing, without causing?

If NONE of T1/T2/T3 fires → the input is COPY. Go to Rule 3 and stop there.
If ANY fires → the input is REPAIR. Go to Rule 4 and continue through Rule 16.
When in doubt about whether a test fired, it did not fire: take the COPY path.

RULE 3 — THE COPY PATH: EMIT THE INPUT CHARACTER FOR CHARACTER, THEN STOP.
Do not add an article. Do not remove an article. Do not swap a verb. Do not swap a noun. Do not add a colour. Do not change a preposition. Do not fix capitalisation, punctuation, or spelling. Do not shorten it because it looks wordy or lengthen it because it looks terse. Copy it and stop.
This path deliberately covers every input register that measured at or near its own ceiling, including the ones that look broken:
  - fluent imperatives with articles ("Place the cap onto the container.");
  - article-free telegraphic shorthand ("put pan from stove to sink", "flip cup upright", "carrot on plate", "open fridge");
  - captions that are not imperatives at all — past tense, third person, or gerund ("moved the silver pot to the left side of the table", "opens the drawer.", "folding cloth from bottom to top left"), and bare fragments ("doing nothing", "end effector reaching banana");
  - glued or opaque object tokens (redcube, stuffedduck, large4fbox, brown1fbox) — do NOT un-glue them;
  - misspellings (banner/bunner for burner, bowel, folk, sliver, yelow, righ, clothe) — do NOT correct them;
  - instructions in French, Spanish, Portuguese, or any language you cannot parse — do NOT translate them.

RULE 4 — THE REPAIR PATH: EDIT THE INPUT, DO NOT COMPOSE A NEW SENTENCE.
Start from the incoming instruction and apply ONLY the deletions licensed by Rule 8 and the one-for-one replacements licensed by Rules 5, 12 and 13. You are a filter, not an author. Never write a sentence from scratch and never restate the task in your own words.
SAFETY VALVE: if, after working through Rules 5-13, you have deleted nothing and replaced nothing, emit the incoming instruction verbatim.

RULE 5 — NAME EVERY OBJECT; NEVER DESCRIBE ONE. (The largest lever in the book.)
For each object or place the instruction refers to, replace any description with a plain name, taking the name from the first source below that supplies one:
  (a) a plain concrete noun ALREADY present in the incoming instruction — if the input already names the thing, that noun is final, even if it is unfamiliar, opaque, a brand word, or a misspelling ("keyboard", "wheel", "coke", "eggplant", "redcube", "bowel");
  (b) the exact name the scene trace gives that object;
  (c) the plainest concrete noun already inside the description itself ("the transparent drinking vessel" → "the vessel");
  (d) if none of (a)-(c) yields a noun, leave that noun phrase exactly as the input wrote it.
When you replace, replace the WHOLE referring phrase — the description and every adjective inside it — with "the" plus the name. Take only the noun from the trace; never harvest adjectives from it (see Rule 10).
  "the metallic vessel typically utilized for boiling liquids" → "the pot"
  "the hand-held implement primarily used for severing ingredients" → "the knife"
  "the black peripheral device used for typing" → "the keyboard"
  "the protruding sliding storage cavity" → "the drawer"
  "the elongated orange taproot" → "the carrot"
Never invent a name. Never guess.

RULE 6 — NEVER SWAP ONE PLAIN NOUN FOR ANOTHER PLAIN NOUN.
If the input already names an object with an ordinary noun, that noun is frozen. Do not "improve" it, however common the alternative sounds, and do not let the trace override it. Forbidden in both directions: plate↔dish, basket↔bin, cloth↔towel↔napkin↔rag, bowl↔cup, cup↔mug↔glass, pan↔skillet, sink↔basin, wheel↔tire↔rim, cube↔block, box↔container, can↔tin, coke↔cola↔soda↔pepsi, keyboard↔keys↔mouse pad. Never generalise a specific token away ("coke can" never becomes "can" or "soda can"). Rule 5 replaces DESCRIPTIONS with names; it never replaces one name with another.

RULE 7 — NEVER INTRODUCE A CATEGORY NOUN AS A NAME.
Never write any of these as an object's name unless that exact word was already in the incoming instruction: object, item, thing, stuff, vessel, receptacle, container, implement, utensil, device, apparatus, entity, unit, piece, element, article, component, specimen.
If the input's own head noun is one of these and Rule 5 finds nothing better, KEEP the input's word — that is the correct answer, not a guess at a specific name.

RULE 8 — DELETE THE WRAPPER. THIS IS THE ONLY DELETION YOU MAY MAKE.
Delete whole spans from this closed list, changing no surviving word:
  8a POLITENESS / ADDRESSEE — please, kindly, thanks, thank you, for me, just, I would appreciate, I would be grateful, I would be obliged, I need you to, I want you to, could you, would you, can you, go ahead and, make sure, be sure to, see to it that, your objective is to, it is imperative that, and any "you" / "your".
  8b CONDITION — spans opening with if, assuming, provided that, should you, in the event that, given that, once you, notwithstanding.
  8c PURPOSE / RESULT — spans opening with in order to, so that, so as to, such that, thereby, for the purpose of, with the aim of, ensuring that, until, which means.
  8d MANNER / DEGREE ADVERBS — carefully, gently, slowly, softly, smoothly, neatly, steadily, firmly, securely, precisely, exactly, deliberately, meticulously, delicately, cautiously, directly, squarely, entirely, completely, fully, properly, definitively, forthwith.
  8e SCENE COMMENTARY — spans describing the scene's current state rather than commanding: which is currently…, that is presently situated…, which currently has nothing on it, is currently vacant, at this moment, regarding the….
  8f ROBOT-BODY TALK — your manipulators, your end effector, your appendage, your gripper, your operational parameters, your processing queue.
Stop deleting when no listed span remains. If deleting a span would leave an ungrammatical fragment, leave that span in place rather than repairing it with new words.

RULE 9 — NEVER DELETE A CONTENT WORD. THIS RULE OUTRANKS RULE 8.
Every noun, adjective, and verb that carries the action must survive. Specifically:
  - never drop the object, the destination, or the source;
  - never drop an adjective the input attached to a named object (keep "silver pot", "blue figure", "small spoon", "yellow basket", "coke can" exactly as given);
  - never drop an alternative in a disjunction — "pot or pan" stays "pot or pan";
  - never drop, merge, split, add, or reorder an action clause. If the input commands two actions, emit two, joined by a bare "and", in the input's order. If the input commands one, emit one.
Rule 8 removes framing. It never removes payload. Where Rules 8 and 9 collide, Rule 9 wins.

RULE 10 — NEVER ADD A CONTENT WORD.
Every noun, adjective, and adverb in your output must already appear in the incoming instruction, or be a name licensed by Rule 5(b)/5(c), or be a verb licensed by Rule 12. In particular, never introduce:
  - a COLOUR (blue, yellow, red, green, orange, white, purple, black, pink, brown, gray, silver …) — except under Rule 11;
  - a MATERIAL (metal, metallic, steel, stainless, plastic, wooden, wood, rubber, ceramic, glass);
  - a SIZE (small, little, big, large, tiny, long, tall, thin, short, bigger, smaller);
  - a SPATIAL word (left, right, upper, lower, top, bottom, middle, center, centre, corner, side, front, back, near, far) — even one the trace shows to be factually correct;
  - an ORIENTATION word (upright, flat, down, all the way);
  - any landmark, brand, or extra object the input did not mention.
If the input already contains such a word, keep it exactly.

RULE 11 — THE ONE PERMITTED ADDITION: A DISAMBIGUATING COLOUR.
You may add exactly ONE colour word, and only when ALL of these hold:
  (i) the trace lists TWO OR MORE items whose head noun is the same as a noun in the instruction (two cans, two cups, two cubes);
  (ii) the instruction does not already distinguish them;
  (iii) the trace makes it unambiguous which of them the instruction means, and gives that one a colour.
Write the colour as one plain word from: blue, yellow, red, green, orange, white, purple, black, pink, brown, gray, silver. Collapse any compound or exotic colour to the nearest word on that list (teal-green → green, off-white → white, crimson → red, brown-coloured → brown). If any of (i)-(iii) fails, add nothing.

RULE 12 — VERBS: KEEP A PLAIN VERB; MAP AN ORNATE ONE.
Keep the input's verb unchanged if it is one of: put, place, set, take, pick up, get, grab, move, push, pull, open, close, shut, fold, unfold, lift, drop, slide, turn, flip, remove, stack, lay, wipe, hold, zip, unzip.
Only if the input's verb is NOT on that list, replace it using this table and change nothing else:
  relocate, transfer, convey, transport, translate, displace, deposit, situate, position, install, reposition, arrange, deliver, settle, stow, insert, place within → put
  retrieve, procure, acquire, apprehend, grasp, secure, obtain, collect → take
  elevate, hoist, raise → lift
  actuate, unseal, swing open → open
  seal, manipulate shut, exert inward force → close
  extract, withdraw, separate, detach → take
Never invent a compound verb, never chain two verbs where the input had one, and never turn one clause into two. Do NOT swap among the plain verbs — put, place, and set measure as identical, and every plain-verb pair is inside noise.

RULE 13 — LEAVE ARTICLES, PREPOSITIONS, CASE, AND PUNCTUATION ALONE.
Do not add or remove "the" or "a". Do not change on ↔ onto, in ↔ inside, or on ↔ on top of in either direction. Do not capitalise, lowercase, add a full stop, or remove one.
Map only ornate prepositions onto plain ones: atop / upon → on; within / within the confines of / into the interior of → in; in the direction of → to.
ONE permitted preposition edit: if the instruction reads "<verb> … to the <container>" where the container is one of pot, pan, bowl, cup, box, basket, drawer, sink, bin, jar, tray, bucket, bag, colander, container — write "into the <container>".

RULE 14 — REQUIRED OUTPUT SHAPE.
The line must be a verb-first imperative naming the object before the destination. Never emit: a question or a question mark; a negation or a "no longer" goal state; a first-person statement of need; a theme-first state description ("The plate needs to have the bowl set on top of it"); a past-tense or gerund opener that the input did not already have; a hyphenated or stacked compound modifier (collapse it to its single plain word: "blue-handled" → "blue", "black rubber" → "black"); or a mention of the gripper, arm, or end effector.

RULE 15 — LENGTH IS A SYMPTOM, NOT THE TARGET.
After Rules 5-14 a repaired line usually lands at 4-12 words. If yours is longer, go back and find the Rule 8 wrapper you failed to delete. NEVER cut a content word, an object name, or an action clause to meet a word budget — once naming and content are correct, word count itself costs essentially nothing, while a dropped content word is the single most expensive error available to you.

RULE 16 — FINAL CHECK, THEN EMIT.
Read your one line and confirm, in this order:
  1. One line, no quotes, no explanation, no reasoning. (Rule 1)
  2. Every noun, adjective, and adverb in it appears in the input, or is a Rule 5 name, or is the Rule 11 colour.
  3. No content word from the input is missing; every clause, every disjunct, every source and destination survives, in the input's order. (Rule 9)
  4. No colour, material, size, spatial, or orientation word that the input lacked (Rule 11 aside). (Rule 10)
  5. No word from the Rule 8 delete lists remains; no category noun from Rule 7 that the input did not have.
  6. Verb-first imperative; no question, no negation, no first person. (Rule 14)
  7. If the input took the COPY path, the output is character-for-character identical to it. (Rule 3)
If any check fails and you cannot fix it by deletion alone, emit the incoming instruction verbatim.

===RATIONALE===

All figures below are within-task deviations in the frozen gripper-reward proxy's logit units, computed from evidence.csv (4,544 phrases, 220 tasks) unless marked "rollout", which comes from the two measured field-note files. The proxy ranks phrases within a task; it is never a success rate. The base_kind column is empty in this diet, so "regime" means the kind label: adversarial (911 ornate rewordings, mean −1.61 within task) vs natural/search/original (3,408 phrases, mean +0.39). Median within-task spread is large, so a tenth of a logit is small and a full logit is decisive.

R1-R3, why triage is the highest-value decision. The task's own canonical wording is rank 1 in 182 of 213 tasks. Of 4,022 measured rewrites, 107 beat their canonical — 2.7% — spread over just 31 tasks, with a median winning margin of +0.05. Splitting by the canonical's register makes the picture sharper: for the 37 tasks whose canonical carries articles, 0 of 672 rewrites beat it (median rewrite −2.03); for the 176 telegraphic tasks, 107 of 3,350 beat it (3.2%, median −0.84). Editing a plain input is a losing bet either way.
The triage thresholds are calibrated, not guessed. Adversarial phrasings run 14-46 words (mean 26.9); canonical instructions run 2-15 (mean 5.4). A ">14 words" trigger catches 99.9% of adversarial phrasings while flagging 0.5% of canonicals (1 of 213) and only 8.5% of fluent natural rewordings; ">12" would have flagged 21% of naturals for no extra adversarial recall. T2 and T3 exist for a short ornate input that the length test would miss.
Rule 3's odd-looking inclusions are measured, not stylistic. Glued canonical tokens are near-ceiling: across 8 tasks whose canonical contains a glued token (close brown1fbox flap, open small4fbox flaps, …), the canonical is rank 1 in 7; the fluent un-glued rewrites lose 0.09 to 0.80. Article insertion into telegraphic input is a no-op — 69 clean single-token "the"/"a" insertions average −0.038 with only 30% helping. So both directions of tidying are forbidden.

R4-R7, why naming is the repair and the only repair that matters. Content-word retention dominates everything. In the adversarial regime: retain ≤25% of the task's content words → −1.97; 25-50% → −1.10; 50-75% → −0.57; ≥75% → −0.08. Regressing adversarial deviation on length, retention, and each stylistic feature, the retention coefficient is +2.34 — six times the next largest term. In the non-adversarial regime the same ladder runs −0.47 (≤25%) to +1.01 (100%). Rollout corroboration: an unfamiliar keyboard named 23%, described as "black peripheral device used for typing" 0.3%.
Rule 6 (never swap a plain noun) resolves the sharpest disagreement among the candidate books. A noun-preference table (plate over dish, cube over block, coke over soda) looks supported until the pairs are traced to their tasks: dish→plate is +0.12 (n=12) but two of its twelve pairs are −0.82 and −0.67; cloth→towel is +0.12 (n=10) with 40% of pairs negative; and the rollout table gives plate over dish 30.5pp on the coke board and exactly 0.0pp on the carrot board. Where a swap does pay, it pays because it moves toward the wording that task itself uses — move→put is +0.23 over 12 pairs, and 11 of those 12 tasks have "put" in their own canonical. An applier never sees the canonical, so it cannot exploit this; the executable form is "keep the input's noun." Rollout losses from getting it wrong are the largest in the corpus: basket→bin 38.9pp, wheel→rim 33.3pp, plate→dish 30.5pp, dropping "coke" 30.5pp, bowl→cup 16.7pp, and a familiar-sounding wrong name ("mouse pad" for keyboard) 3% against 18-23%.
Rule 7's fallback repairs the previous rule set's worst observed failure, which emitted "the purple object" when its trace failed to name an eggplant. Introducing a category noun measures −0.42 in the non-adversarial regime (−0.00 vs +0.42, n=34) and −0.28 inside the adversarial regime; keeping the input's own "object"/"thing" costs nothing, since both are ordinary corpus words (object 1,553 uses, thing 320).

R8-R9, what the wrapper is worth and where deletion must stop. Inside the adversarial regime, controlling for length and retention: a purpose/result clause −0.38 (n=98), a manner adverb −0.18 (n=43), politeness +0.01 (n=252), periphrasis only −0.10 once retention is held fixed — confirming that ornate description harms through naming, not through ornateness. Politeness is genuinely harmless: 9 clean single-word pairs give please→∅ = +0.27 (9 of 9 helped) and the rollout single-edit table gives please→∅ = 0.0pp. It is deleted because it is free and shortening, never as a priority, and never spent as an edit on a COPY-path input.
Rule 9 outranks Rule 8 because losing a content word is the most expensive move available. Non-adversarial ladder by count of the task's content words missing: 0 → +1.01, 1 → +0.58, 2 → +0.14, 3 → −0.19, 4 → −0.72; a within-task regression controlling novelty and length puts it at −0.26 per missing word against −0.18 per added word and +0.02 per word of length.

R10-R11, why additions are banned and where the single exception sits. Holding the task fixed, on tasks whose own instruction lacks the class, adding one costs: colour −0.53 (+0.06 vs +0.59, n=938), material −0.69 (−0.23 vs +0.46, n=208), size −0.63 (−0.20 vs +0.43, n=65), spatial −0.42 (+0.03 vs +0.45, n=175). Clean single-token insertions agree in sign and rank: +size −0.68 (n=12, 33% helped), +material −0.27 (n=22, 18% helped), +silver −0.34, +metal −0.42, +small −0.81 (worst: "grab the metal pot from the stove" → "grab the small metal pot from the stove", −4.60). Rollout: adding "yellow" −19.4pp, small→teal −33.4pp, adding "rubber" −13.9pp, and accurate left/right qualifiers taking a task from 65% to 6-10% — accuracy is not the issue.
Rule 11 survives because the sign genuinely flips under ambiguity, and the flip is measurable in this diet. On the 7 tasks whose name marks a distractor scene, phrases adding a colour the canonical lacks score +0.33 against −0.00 for those that do not (n=71); on every other task the same edit scores +0.05 against +0.56 (n=942). The exception is fenced by three conditions the applier can check against the trace, with "add nothing" as the default, because a colour added to the wrong item is exactly the −19.4pp case.

R12-R15, the small stuff, sized honestly so it is not over-applied. Verb identity carries about a third of a logit and only in one direction: a phrase using the task's own verb scores +0.61 against +0.29 for one that swaps it (n=962 / 1,549). Among plain verbs there is nothing to win — put→place +0.006 (n=87), put→set +0.001 (n=35), set→place +0.001 (n=28), and the rollout board gives set→put exactly 0.0pp — which is why Rule 12 forbids spending an edit there and repairs the previous book's mistaken ban on "set". Ornate→plain mappings do pay: open up→unfold +0.87 (4 of 4), lift→take +0.30 (4 of 4), grab→pick up +0.22 (n=16).
Prepositions and articles are noise: on→onto +0.014 (n=28, 50% helped), in→inside −0.011 (n=19), onto→on top of −0.06 (n=7), article insertion −0.038 (n=69). "on top of" measured +8pp on the spoon/towel rollout board and −22.2pp on the coke/plate board, so it is protected in both directions rather than banned. The single exception, to→into for a container goal, is +0.21 with 6 of 7 pairs positive.
Rule 15 corrects the most common wrong lesson in this evidence. Raw length correlates strongly with score, but the correlation is content-loss wearing a disguise: hold retention at 100% and novel content at ≤1, and mean deviation across length bins is +1.30 (≤5 words), +1.19 (5-7), +1.24 (7-9), +1.03 (9-12), +1.58 (12-16) — flat. The controlled length coefficient is +0.02 per word in the non-adversarial regime and −0.026 inside the adversarial band. A hard word ceiling would therefore buy nothing and would tempt the applier into the −0.26-per-word deletion error, so the budget is written as a diagnostic.
Rule 14's bans come from the bottom of the boards: negation −2.08 (n=3), gerund openers +0.01 against +0.42, hyphenated compounds −0.17, and the worst-ranked fluent rewrites in the whole set are theme-fronted state descriptions. The corpus itself contains no negation, no questions, no second person, no manner adverbs, and no politeness above a floor of 5 occurrences in 17,297 instructions, and is verb-initial in over nine lines in ten.
One contrary signal deliberately not acted on: question and modal frames score +0.55 against +0.40 in the non-adversarial regime (n=356). Acting on it would mean writing forms the corpus never contains, and the one member of that family with a rollout number ("please") came back at exactly 0.0pp. Rule 14 bans them rather than chasing an unreplicable proxy gain.
Rule 1 has no measurement behind it; it is the precondition for any of the rest being applied at all by three different rule-followers, one of them 9B.