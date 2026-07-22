# Phrasing rules v2 for the instruction rewriter

Authored from `results/analysis/b4_rules_inputs/` (files 01–09). File 08
(measured rollout search, n-labeled pairs) is the highest authority; where it
conflicts with files 03–07 or with corpus statistics, file 08 wins. This
revision keeps v1's load-bearing structure — trace-noun adoption, description
deletion, plain register, length budget — and repairs the seven failures in
file 09.

**Inputs available at rewrite time:** the incoming instruction, the trace
(scene sentence + description→plain-name mapping, file 07 format), and the
training corpus (files 01/02). No rule below requires anything else.

**Evidence discipline (repairs v1 failure 7).** v1 banned tokens on the
strength of confounded pairs (it blamed "set" for the harm done by "exactly
in the middle", 03 spoon pairs). v2 cites PURE single-edit pairs from file 08
wherever one exists, and treats multi-edit deltas and file-05 pool winners as
structure hints only (08 "Reading discipline"; 05 CAUTION note; winner's-curse
entries in 08 scenes 2, 3, 6, 7).

**Procedure.** Apply the rules in order: (1) resolve every object reference
to a concrete noun (R1–R4), (2) choose adjectives (R5–R6), (3) choose
relation and frame (R7–R8), (4) delete and trim to budget (R9), (5) stop at
the minimal edit (R10).

---

## R1. Adopt the trace's noun; rename only for visual fidelity

**Rule.** Replace every object description with one concrete noun. Default is
the trace mapping's plain name. Override it in exactly one case: the mapped
name is rare or absent in the training corpus (check file 01) AND the trace's
own scene sentence supplies a familiar basic-level noun that literally names
what the object looks like — then use that familiar noun. If no familiar noun
fits the appearance, keep the object's true name even though the corpus lacks
it. Never rename to a noun that could pick out a different object in the
scene, and never drop one of the task's two objects — both the thing moved
and the destination must be named.

**When it applies.** Every rewrite; it is the core trace-noun-adoption step.

**Worked example.** Incoming: "nest the citrus fruit inside the small fluted
glass votive holder." Trace maps "small fluted glass votive holder" →
"votive holder"; scene sentence says it "looks like a small glass cup."
"votive holder" is corpus-absent, and the scene supplies the look-alike noun
— rewrite: "put the orange in the glass cup." Counter-case: the destination
is a stapler; nothing familiar looks like a stapler, so keep "stapler".

**Evidence.** 08 scene 1 (ramekin phrased as "white bowl": +63.9pp verdict;
"bowl→ramekin true name" −34.8); 08 scenes 3–4 (keyboard and wheel kept by
every winner; renaming unavailable); 08 cross-cutting "visual-fidelity
naming" (replicated ×3); 06 Cases 2–4; 04 ("above the mouse pad" — a familiar
noun for the WRONG object — 3.1%); 08 scene 2 ("stack the cubes", one object
unnamed, 0.0% vs 27.1%).

## R2. Category words are banned — with a fallback ladder for failed traces (repairs v1 failure 1)

**Rule.** Never emit a superordinate/category word for an object: *object,
thing, item, element, container, vessel, device, shape, one*. This holds even
though such words are frequent in the corpus (file 01: "object" on 1,527
lines) — measured harm overrides corpus frequency. If the trace fails to map
an object (mapping missing, or maps to a category word), fall back in order:
(a) take a concrete noun for that object from the trace's scene sentence;
(b) else commit to the single most likely basic-level noun implied by the
incoming description itself; (c) else keep the description trimmed to its
color word plus its most concrete head noun (≤4 words). A trimmed description
is a survivable worst case; a category word is catastrophic.

**When it applies.** Whenever a trace mapping is absent or vague — the case
v1 had no answer for.

**Worked example.** Incoming: "move the shiny wire kitchen tool onto the
folded napkin"; the trace maps it only to "the metal object". Scene sentence:
"a metal whisk lies beside the stove." Fallback (a) gives "whisk" — rewrite:
"put the whisk on the napkin." Had the scene sentence also failed, (b) still
commits to "whisk" from "wire kitchen tool"; only if no noun can be committed
does (c) allow "the shiny wire tool" — never "the metal object".

**Evidence.** 06 Case 2 ("...place it in the white object" 4% vs rich
description 42% vs "bowl" 64%); 09 failure 1 (v1 emitted "the purple object",
−31pp class); 03 (category-worded phrasings at 0.1–4.9%: "lush green
element", "carbonated drink container", "circular object used for vehicle
traction"); 05 (every category-worded draw at or near 0%).

## R3. Exact noun identity: most-specific fitting noun, no cover terms, no synonym swaps (repairs v1 failure 2)

**Rule.** Near-synonyms are cliffs, not shades. Among candidate concrete
nouns for an object: (a) prefer the most specific basic-level noun that
literally matches the object's visible shape — a literal shape-noun beats a
looser conventional name; (b) never widen to a cover term that spans several
object shapes (the genus word that could equally name two different
household things); (c) never swap the trace's noun for a synonym — if the
trace's noun is concrete and corpus-attested, keep that exact token. Corpus
frequency (file 01) is only the tie-break between nouns that are equally
faithful to appearance.

**When it applies.** Whenever more than one concrete noun could name the
traced object.

**Worked example.** The traced destination is a woven straw container. The
candidates are the specific noun a glance suggests and the storage cover
term ("hamper" vs "receptacle"/"storage"): choose "hamper". If the object
moved is a perfectly rectangular bar of soap described as "the white brick-
shaped bar", say "bar" (the literal shape noun), not "toiletry".

**Evidence.** 08 single-edit table: basket→bin −38.9, plate→dish −30.5,
bowl→cup −16.7 (all PURE); towel→cloth −13.9; 08 scene 2 cube→block +15.1
with scene reading "literal, perfect cubes — visual-fidelity naming"; 08
cross-cutting "Exact-token identity: near-synonyms are cliffs" (replicated
×5); 06 Cases 1 and 7; 09 failure 2 (v1 chose "dish" and "block").

## R4. Brand tokens are load-bearing — never genericize them (repairs v1 failure 4)

**Rule.** If a brand or product name for an object appears anywhere in the
incoming instruction or the trace, carry that exact token into the rewrite,
even if the trace's mapping offers a generic name and even if the brand token
never appears in the training corpus. Do not replace it with the product
category. (Getting the brand slightly wrong is measured as nearly free;
deleting it is expensive — so when in doubt, keep it.)

**When it applies.** Branded packaged goods: drink cans/bottles, labeled
boxes, toys.

**Worked example.** Incoming: "shift the orange soft-drink bottle next to the
sink onto the tray"; trace maps it to "soda bottle" but the incoming text
identifies it as a Fanta. Rewrite: "put the fanta bottle on the tray" — not
"the soda bottle".

**Evidence.** 08 scene 5 PURE single edits: coke→∅ −30.5, coke→cola −16.7,
coke→soda −13.9, coke→pepsi ≈ free (−2.8); 09 failure 4 (v1's "soda can" for
a coke can; the brand token measures +14 to +31pp over soda/cola/bare-can).

## R5. Adjective budget: at most one plain color word, and only where it earns its keep

**Rule.** Each noun gets at most one adjective, and it must be a basic color
word. Add it only if (a) the noun is rare or absent in the training corpus
(the color anchors an unfamiliar word), or (b) the scene contains another
object the bare noun could match. Otherwise use the bare noun — decorating a
well-known target costs success. Never use material or texture adjectives
(*ceramic, rubber, metal, plastic, wooden*), and never adverbs of manner or
size qualifiers.

**When it applies.** Every noun phrase, after R1–R4 fix the noun.

**Worked example.** Destination is a corpus-alien "router": write "the black
router" (color supports the OOV noun). Object is the only pan in the scene:
write "the pan" — not "the steel pan" (material) and not "the large gray pan"
(stacked adjectives).

**Evidence.** 08 cross-cutting "Adjectives help only where the task is
hard/OOV; hurt on good-nominal tasks" (replicated ×4: +black with keyboard
and wheel; PURE −13.8 for +yellow on plate, −19.4 for +blue on towel despite
"blue towel" ×925 lines in file 01); 08 cross-cutting "Color ≫ material"
(PURE −13.9 for +rubber; −37.0 white→ceramic multi-edit) plus corpus template
(file 01: color-adj+noun combinations in the thousands, "ceramic" 0,
"rubber" 10); 03 (adding "orange"/"black" to carrot/tire pairs +9 to +12pp).

## R6. Simplify every color to a basic corpus color word (repairs v1 failure 3)

**Rule.** When a color adjective survives R5, map it to the nearest member of
the corpus's high-frequency color set (file 01: blue, yellow, red, green,
orange, white, purple, black...). Compound, hedged, or precise color terms
(teal-green, yellowish-orange, light green, burgundy, off-white) must be
simplified to their basic neighbor, even when the precise term is the more
accurate description and even when the trace uses it verbatim.

**When it applies.** Any color word taken from the trace or the incoming
instruction.

**Worked example.** Trace describes "a burgundy-ish red mug". If R5 licenses
a color at all, write "the red mug" — never "the burgundy-ish red mug".

**Evidence.** 08 scene 2 (PURE single edit: teal-green→plain +14.6; cleanest
table +33.4 for small/plain-green over teal) with scene reading "the honest
'teal-green' is corpus-alien"; 09 failure 3 (v1 copied the trace's color
verbatim, −30pp); file 01 counts (teal 0, yellowish 0, "light green" 6 vs
green 1,670, blue 2,680).

## R7. One plain relation word that matches the geometry; no precision, no orientation (repairs v1 failure 6)

**Rule.** Express the placement with a single plain preposition chosen by the
geometry the trace describes: containment (destination is hollow and the
object goes inside) → "in" / "into" / "inside"; support (object rests on a
surface) → "on". This overrides the incoming instruction's own preposition.
Do not add: "on top of" (default against it), centering phrases ("in the
middle/center of", "exactly..."), side qualifiers ("on the left", "to its
right") even when accurate, or orientation words ("upright", "vertically") —
v1 required "upright" for containment; measured evidence says drop it.

**When it applies.** Every rewrite, when composing the placement clause.

**Worked example.** Incoming: "insert the tennis ball so it rests centered
upright inside the cooking pot on the right." Rewrite: "put the ball in the
pot."

**Evidence.** 08 scene 1 (in-bowl vs on-ramekin +37.0; PURE ±upright +7.8;
09 note: orientation words pooled ~+8–10pp for dropping); 08 scene 5 (PURE
"on top of" −16.6); 03 spoon pairs ("exactly in the middle" −31 to −36pp;
"in the center of" variants consistently below plain "on"); 06 Case 6
(accurate left/right detail, −55 to −64pp); 04 (every "center of the tire"
row at 0–3% vs "on the black tire" 29.2%); 08 scenes 3–4 whole-name-over-
part-name (keys −23.6, rim −29.2 PURE — a part-name is a hidden precision
target; name the whole object).

## R8. Verb and frame from a short whitelist; "set" and "please" are legal (repairs v1 failure 5)

**Rule.** Output exactly one imperative sentence in one of two frames:
single-clause "put/place/set the X on|in the Y", or two-clause "pick up the
X and put/place/set it on|in the Y". "set" is a full peer of "put"/"place",
and a leading "please" is harmless — neither is worth an edit to remove
(v1 banned both; measurement cleared both). Avoid the verbs *move, take,
grab, retrieve, arrange, position, shift, relocate, transfer, insert,
balance* and the frame "move/take X to Y" (it hides the geometry that R7
must express). File 02 shows the policy saw those fancy verbs as training
paraphrases; measured rollouts show they still lose — training exposure does
not make a verb safe.

**When it applies.** Every rewrite.

**Worked example.** Incoming: "Kindly relocate the mug across to the tray
region." Rewrite: "place the mug on the tray" (or "set the mug on the tray"
— equally legal; and if the incoming had begun "please", keeping it costs
nothing).

**Evidence.** 08 scene 3 (PURE set→place +5.1; "set" is the confirmed board
winner), scene 5 (set ≈ put within noise; PURE "please" ≈ free, +5.6), 08
cross-cutting ("set ≈ put, please free — both wrongly banned by rules-v1",
replicated 4+ boards); 08 scene 1 (place→move +35.2), scene 5 (pick-up→take
+25.0); 03 ("Arrange..." rows 4.9–62%, always the losing side; "Move the
spoon to the center..." −26pp class); 06 Case 7 ("arrange...neatly" −40pp);
09 failure 5.

## R9. Length budget with a floor: a short full sentence, nothing extra

**Rule.** Target ≤10 words, hard cap ~12 (corpus median 10, p90 14 words,
file 01); one sentence; at most the two clauses of R8. Delete: manner adverbs
(*carefully, gently, neatly, securely, exactly*), scene context, spatial
side-detail, justifications, and any adjective R5 did not license. But do not
cross the floor: keep the articles and the imperative verb — telegram style
("X in Y") measurably loses to the full short sentence.

**When it applies.** Final trim of every rewrite.

**Worked example.** Incoming: "The pan sitting at the left edge of the
counter needs to be moved over so that it sits gently on the folded napkin."
Rewrite: "put the pan on the napkin" — not "pan on napkin".

**Evidence.** 06 Case 6 (verbose accurate phrasings 1–10% vs 65% nominal);
03 (the two worst phrasings in the file are the two longest, 1.0–2.8%;
corpus-absent adverbs: "exactly" 0, "carefully" 0 in file 01); 08 scene 8
(full sentence vs telegram +34.2); 08 scene 2 ("stack the cubes"
over-trimmed to 0.0%).

## R10. Minimal edit — stop when it looks like the corpus

**Rule.** If applying R1–R9 to the incoming instruction yields a phrase that
already reads like a training-corpus line (short plain imperative, concrete
attested nouns), output it and stop. Do not embellish, do not chase a fancier
candidate, and when several legal phrasings remain, choose the one closest in
wording to the incoming instruction's task content. Plain-and-attested is the
ceiling far more often than not.

**When it applies.** Always — it is the stopping criterion.

**Worked example.** Incoming: "set the fork on the tray." Already legal under
every rule (whitelisted verb, concrete nouns, plain relation, under budget):
output "set the fork on the tray" unchanged.

**Evidence.** 08 cross-cutting "Good nominal ⇒ unbeatable" (held-out crowned
the plain nominal 4/4) and "Winner's curse universal" (every search-split
leader regressed; embellishments like "+colors" and "on top of" collapsed on
held-out layouts, scenes 6–7); 06 Case 7 (the minimal cleanup edges the
nominal); 05 (in every pool, the winning draws are the plainest frames — use
as structure only).

---

## Known tensions and how to arbitrate

1. **Rename vs keep the odd name (06 Case 2 vs Case 3; 08 scene 1 vs scenes
   3–4).** Renaming rescued the ramekin (+63.9) but no substitute helped the
   keyboard or wheel. Arbitrate with R1's test: rename only when the trace's
   scene sentence itself supplies a familiar noun that literally names the
   look ("is a white bowl to the eye"). If you must reach or reason to find
   the substitute, don't — keep the true name and let R5 add one color word
   (the measured keyboard/wheel solution). A wrong-object rename is the worst
   outcome of all (04: "mouse pad" 3.1%).

2. **Corpus frequency vs measured preference.** "object" (1,527 lines) and
   "move" (7,124 lines) are corpus-frequent yet measured toxic; "coke" and
   "keyboard" are corpus-absent yet measured best. Rule: file 08's measured
   token effects override corpus statistics wherever they speak; corpus
   frequency is only (a) the R1/R5 rarity test and (b) the R3 tie-break
   between equally appearance-faithful nouns.

3. **"on top of" (08 scene 5 vs file 04 spoon rows).** PURE −16.6 on one
   scene, +8.3 (n=288, near noise) on another. The harm is measured cleaner
   than the help, and the help did not replicate — default to plain "on"
   (R7) and accept the occasional foregone point.

4. **Adjective on the destination (08 cross-cutting).** +black rescued hard
   OOV targets; +yellow/+blue hurt well-named ones (PURE −13.8). The R5 test
   (corpus-rare noun or ambiguous scene → one color word; else bare) is
   exactly the measured boundary; when genuinely unsure, prefer bare — the
   downside of decorating a good noun is measured, the upside is not.

5. **One clause vs two (08 scene 1 vs scenes 5–8).** Dropping "pick up ...
   and" cost 7.4pp on the hardest scene but the single-clause nominal frame
   won everywhere the nominal was good. Both frames are legal under R8;
   pick whichever is the smaller edit from the incoming instruction (R10),
   leaning two-clause only when the placement is containment in an OOV
   receptacle (the one measured case it helped).

6. **Trace's generic mapping vs richer tokens elsewhere (09 failure 4).**
   The trace mapping is authoritative for WHICH object is meant (R1), but
   not for the final token: a brand token (R4) or a more appearance-faithful
   specific noun (R3) found in the incoming instruction or scene sentence
   overrides the mapping's generic name.

7. **File 05 pool winners.** Never cite a single high-scoring draw as
   proof a phrase is best (SE ~10pp at n=24; oracle flags are selected on
   noise). Use pools only to confirm frame structure — which these rules
   already encode from files 08/03/04.
