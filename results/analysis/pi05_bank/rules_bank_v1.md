===RULES===
1. Copy verbatim, into your rewrite, the head noun of every thing the instruction tells the robot to act on and the head noun of every place it tells the robot to put that thing — the last noun of each referring phrase ("bowl", "book", "compartment", "caddy", "stove", "drawer", "cabinet") — never dropping one and never swapping one for another word, however synonymous.
2. Copy verbatim every spatial or relational word the instruction uses — left, right, middle, front, back, top, bottom, inside, between, next to, side, rear, upper, lower — and do not go looking for a location the instruction did not state.
3. Reuse the instruction's own action verb and its particle rather than a synonym ("push", not "slide" or "shove"; "turn on", not "ignite"), and change the verb only when you cannot make your output differ from the input any other way.
4. Never emit the instruction back unchanged: always alter at least one thing, taking that change from word order, connective and framing wording, or an added scene descriptor, and never from the words rules 1-3 protect.
5. Do not compress or de-formalise to look more like a robot command — keep politeness, question form, first-person framing and extra length if they come naturally, cap your output's length nowhere, and shorten below the input's length never.
6. You may attach a descriptor of colour, material, size or orientation to the instruction's own referring phrase ("the black bowl at the back" -> "the marbled black bowl at the back"), but only a descriptor the scene description actually asserts of that object, and only by adding words, never by deleting any of the instruction's own.
7. As a last tie-breaker between two rewrites you like equally, keep a bare "in" as "in" rather than expanding it to "inside".

===RATIONALE===

All numbers below are my own recomputation from `results/analysis/pi05_bank/distill_evidence/phrases.csv`
(2,587 measured phrasings, 65 tasks, 47,750 real rollout episodes). Unless stated otherwise the
statistic is an **episode-weighted contrast computed inside each (task x generation-tier) cell** and
then averaged over cells. That stratification is load-bearing: the README warns that the
natural/adversarial/oracle_board tiers came from one prompt family, so length, novel-word count,
non-verb-initial phrasing, clause structure and noun-dropping all co-occur *by construction*, and
any contrast pooled across tiers partly measures "which prompt wrote this row". Bracketed intervals
are 2.5/97.5 percentiles of a 2,000-draw bootstrap over cells. Word matching is inflection-tolerant
(bowl/bowls, place/placing/placed count as the same word), so morphology is never scored as a drop.

The corpus is two populations and the rulebook is built for both:
  - 10 in-finetune tasks (`libero_goal`, canonical instruction IS a training string, canonical
    success 96.7%, mean headroom 3.3pp, 11,260 eps). Nothing to gain, up to 60pp to lose.
  - 55 out-of-finetune tasks (`libero_90`, canonical success 37.7%, 36,490 eps). Of these, 27 sit
    below 20% canonical with mean headroom 7.3pp (a floor where nothing moves), 17 sit above 70%
    with 5.8pp, and only 11 sit in the 20-70% band with real headroom (36.5pp, mean best 78.3%).
The rephraser cannot tell which population an instruction belongs to. So the rulebook is written to
be **large and protective where the policy already knows the instruction, and free everywhere else**.
Validation of exactly that: rewrites obeying rules 1-4 beat rule-violating rewrites by +7.34pp within
(task x tier) on the in-finetune tasks (19 cells, 14 positive / 3 negative, [+2.3,+13.5]) and by
+0.24pp out-of-finetune (56 cells, 24 positive / 14 negative); measured against simply passing the
instruction through, they cost -0.75pp in-finetune (10 tasks, median 0.00, 990 eps) and -0.01pp on
the 38 out-of-finetune non-collision tasks (median 0.00, 3,900 eps), while *indiscriminate*
rephrasing costs -16.11pp and -2.00pp respectively.

RULE 1 — head nouns. The single largest and only large effect in the bank.
  In-finetune: dropping or replacing at least one head noun costs **-32.16pp** within (task x tier),
  24 cells, 10 tasks, 21 of 24 cells negative, 8,055 eps (2,550 violating / 5,505 clean),
  [-42.1,-22.9]. It replicates inside each tier computed separately — natural -29.79pp (6 tasks,
  5/6 negative), adversarial -44.83pp (10/10 negative), oracle_board -12.36pp (7 tasks, 5/7) — and
  the dose response is monotone *within* each tier, so it is not tier composition: natural scores
  92.7% / 63.9% / 10.0% for 0 / 1 / 2 head nouns lost; adversarial 75.2% / 36.4% / 12.7%;
  oracle_board 95.8% / 82.4%. Leave-one-task-out over all 10 tasks gives -28.9 to -35.1pp; it never
  collapses onto one task. Out-of-finetune it is free: -1.58pp, 109 cells, 55 tasks, 33 negative /
  34 positive, median exactly 0.0, [-4.0,+0.8].
  The rule protects *head* nouns only, and that scope is measured, not assumed. Changing or dropping
  a pre-modifier while keeping the head noun is null: in-finetune -0.49pp (6 cells, 3 tasks),
  out-of-finetune +0.70pp (48 cells, 24 tasks, 17 positive / 14 negative, [-2.0,+3.4]). This matters
  concretely — the best-known phrasing on libero_90:70 ("put the chocolate pudding to the right of
  the plate", canonical 75.0%) is "On the right side of that plate, please put the brown pudding."
  at 94.3% on n=105, which replaces the modifier "chocolate" with "brown" and keeps "pudding" and
  "plate". A rule protecting every content word would have forbidden it.
  Frequency of violation: an ordinary rephraser breaks this on 21.1% of out-of-finetune and 14.8% of
  in-finetune natural-tier rewrites, and the downside is a lottery rather than a discount — among
  in-finetune rows with one head noun swapped, most of the mass sits far below canonical.
  Isolation checks: with task fixed effects, episode weights, and n_words plus novel-word count as
  controls, noun loss scores -5.85 (t=-5.86) across all 2,587 rows and -26.42 (t=-9.49) in-finetune,
  while n_words is dead (-0.09, t=-0.43). Length and unusual vocabulary are proxies; the noun is the
  cause.

RULE 2 — spatial words. Real, small, and cheap.
  Among rewrites that already keep every head noun, dropping a spatial word costs **-2.89pp**
  out-of-finetune: 56 cells, 32 tasks, 12,530 eps, 19 negative / 14 positive, [-6.0,-0.3]. The sign
  is consistent in all three tiers computed separately (natural -3.93pp over 20 tasks, adversarial
  -1.43pp over 23, oracle_board -5.00pp over 11), and a task-fixed-effects episode-weighted
  regression with length and novel-word controls puts it at -2.35 (t=-2.09) over all rows and -2.03
  (t=-2.08) out-of-finetune. In-finetune the bank cannot measure it (+0.87pp over 10 cells but only
  4 tasks have a spatial word and enough clean rows).
  This is a *retention* rule and nothing more. Introducing a spatial word the instruction lacks is
  null once head nouns are held fixed: +0.48pp out-of-finetune (40 cells, 20 tasks) and +0.36pp
  in-finetune (13 cells, 5 tasks); adding an extra one alongside the originals is -1.53pp (36 cells,
  21 tasks, 15 negative / 7 positive). The clause "do not go looking for a location the instruction
  did not state" is therefore justified as free rather than as beneficial — it buys nothing measured,
  and it removes an opportunity to assert something about the scene that is wrong. Violation rate for
  an ordinary rephraser: 23.9% out-of-finetune, 32.0% in-finetune.

RULE 3 — the verb. A preference, at roughly a third of rule 1's weight and a tenth of its confidence.
  This resolves the corpus's sharpest internal disagreement, so the contrast is stated the strict
  way: among rewrites that *already* obey rules 1 and 2, in-finetune, those that also keep the
  canonical verb land at **-0.75pp** versus the instruction (10 tasks, median 0.00, 990 eps) while
  those that swap it land at **-5.15pp** (10 tasks, median -1.07, 4,390 eps); verb-keeping is the
  better of the two on 6 of 10 tasks and never worse than -4.65pp. The pooled cell contrast agrees
  in sign and size but not in significance: verb loss among head-noun keepers is -2.31pp in-finetune
  (22 cells, 13 negative / 6 positive, [-5.7,+0.4]) and +1.66pp out-of-finetune (64 cells,
  [-1.5,+5.0]) — i.e. a wash out-of-distribution.
  The mechanism is visible on libero_goal:5, "push the plate to the front of the stove": the
  fidelity-preserving rewrites that keep "push" score 50 / 80 / 80 / 97.1 / 100 / 100 percent, while
  those substituting "slide", "shove" or "move" score 20 / 30 / 40 / 50 / 60 / 60 / 60 / 70 / 80 /
  88.6. Because 83.4% of ordinary out-of-finetune rewrites change the verb, a hard ban would be
  expensive relative to a 2-4pp effect whose interval spans zero, which is why the rule is worded as
  "change it last" rather than "never change it" — rules 4-6 leave plenty of other places to spend
  the required edit.

RULE 4 — never echo the instruction. The only generative lever in the whole corpus, and it rests on
  two tasks, so it is worded to be unconditional and free.
  Two out-of-finetune tasks have a canonical instruction that is *verbatim* one of the 40 finetune
  strings, and on both the memorised string fires memorised behaviour in the wrong scene:
    - libero_90:44, "turn on the stove": verbatim 2.0% (n=150) -> rewrites obeying rules 1-3 92.9%
      (6 phrases, n=85); all 45 measured rewrites 91.3% (n=1,100). Headroom 98.4pp, the largest in
      the bank.
    - libero_90:77, "pick up the book and place it in the back compartment of the caddy": verbatim
      18.0% (n=150) -> all 77 measured rewrites 57.3% (n=1,715). No rewrite in the bank obeys rules
      1-3 on this task, so the compliant cell is unmeasured here.
  Graded similarity to a training string does NOT produce this; only exact lexical identity does,
  which is why the rule can only be "always change something" and not "change it when it looks
  trained". The cost of obeying is the reason it ships: -0.75pp in-finetune (10 tasks, median 0.00,
  3 of 10 positive) and -0.01pp across 38 out-of-finetune non-collision tasks (median 0.00, 3,900
  eps). Over the 49 tasks where the bank contains a rules-1-4-compliant rewrite the expected effect
  is +1.70pp per task, against -2.85pp for rephrasing without the rules. Honest tail risk: the worst
  in-finetune task under full compliance is libero_goal:5 at -10.81pp (6 phrases, 185 eps), and 2
  tasks lose more than 5pp.

RULE 5 — do not compress. This rule exists because the obvious doctrine is backwards.
  Every terseness-and-formality constraint I tested is null in-distribution and positive
  out-of-distribution once rules 1-2 are held fixed. Non-verb-initial phrasing: in-finetune -0.49pp
  (19 cells, 10 tasks, [-2.4,+1.3]); out-of-finetune **+2.87pp** (68 cells, 51 tasks, 25 positive /
  15 negative, [+0.6,+5.8]); on the 19 out-of-finetune tasks with at least 15pp of headroom
  **+4.65pp** (44 cells, 30 positive / 12 negative, [+1.0,+9.2]). Politeness: +1.54pp in-finetune
  (20 cells), +1.07pp out (76 cells, 51 tasks). Longer than 1.5x the input: -1.08pp in-finetune
  (16 cells, [-3.2,+0.9]), +0.54pp out (34 cells, 28 tasks). Question form: -2.45pp in-finetune
  (11 cells, 7 tasks, [-5.4,+0.05]), +0.06pp out (31 cells). Word-count residuals within (task x tier)
  are flat from 7 words to 25-plus; the only negative bin is 6 words or fewer (-3.04pp
  out-of-finetune), which is why the rule forbids going *shorter* than the input and caps nothing.
  The concrete stakes: the top scorers on libero_90:44 (canonical 1.65%) are "Could you start the
  burners?" 100.0% (n=90), "I need you to ignite the stovetop." 100.0% (n=90) and "Please switch the
  stove on." 100.0% (n=90); the best phrasing on libero_90:70 is the polite reordered one quoted
  under rule 1 at 94.3% against a 75.0% canonical; the best on libero_90:33 ("close the microwave",
  80.0%) is "Can you close the microwave for me?" at 100%, though on n=10, so read that last one
  as colour rather than evidence. A terse-imperative rule would have forbidden all of them.

RULE 6 — descriptors. Safe, additive, and worth nothing on its own.
  Adding a colour or material word to a rewrite that already obeys rules 1-2: +1.93pp across all
  tasks (63 cells, 40 tasks, 28 positive / 20 negative, [-1.0,+5.0]), -0.23pp in-finetune (20 cells,
  10 tasks) and +2.94pp out-of-finetune (43 cells, 30 tasks, [-1.0,+7.2]). That is a permission, not
  a lever — the interval spans zero everywhere. The apparent upside of *swapping* the instruction's
  own colour word for a different one (+3.17pp residual out-of-finetune, 88 rows, 10 tasks, 1,900
  eps) shrinks to +1.01pp (19 rows, 9 tasks) once the searched oracle_board tier is removed, so it is
  mostly selection.
  The reason the rule insists on scene-attested descriptors is a clean same-task contrast on
  libero_goal:5: "push the red-rimmed plate in front of the stove" scores 97.1% (n=105) while "slide
  the red-striped plate to the front of the stove" scores 20.0% (n=15). Same task, same protected
  nouns; one descriptor is true of the object and the other is not. The bank cannot measure
  descriptor accuracy in general (there is no scene ground truth column), so this is one paired
  example plus the null on adding descriptors — enough to make the permission conditional, not enough
  to claim an effect size.
  "Delete none of the instruction's own words" is not an extra finding; it is rules 1-2 restated
  inside this permission, which is where a rephraser is most likely to violate them.

RULE 7 — "in" stays "in". A tie-breaker, honestly labelled.
  Expanding a bare "in" to "inside" costs **-2.63pp** out-of-finetune: 48 cells, 22 tasks,
  13,655 eps, 23 negative / 9 positive, [-4.65,-0.62]. Restricted to rewrites that keep every head
  noun it is -1.99pp (34 cells, 19 tasks, [-3.74,-0.35]), so it is not just noun loss in disguise,
  and it is negative in each tier computed separately (natural -2.24pp over 22 tasks [-4.5,-0.05],
  adversarial -2.73pp over 17, oracle_board -3.39pp over 9). It is the only lexical-substitution
  effect in the bank that survives tier stratification.
  Do not generalise it. "Into" goes the other way (+1.24pp, 54 cells, 22 tasks, 26 positive / 12
  negative) and blanket preposition preservation is null (out-of-finetune -1.05pp over 81 cells,
  in-finetune +1.29pp over 21 cells). The rule is one word, because one word is all the evidence
  supports. At roughly 2pp it should never override rules 1-6.

WHAT WE TESTED AND REJECTED

  - "Keep the output to one imperative clause of at most 1.5x the input's length." Refuted, and
    backwards. In-finetune, "over 1.5x" is true of 150/150 adversarial rows and 0 others, so the
    pooled contrast is a pure between-tier comparison. Within (task x tier) and among rewrites
    obeying rules 1-2, >1.5x is -1.08pp in-finetune and +0.54pp out; the shortest bin is the only
    negative one. Rule 5 now states the opposite.
  - "Start with the bare imperative verb; never a question or a first-person request." Refuted with
    a sign reversal: +2.87pp out-of-finetune ([+0.6,+5.8]) and +4.65pp on the headroom tasks, -0.49pp
    in-finetune. It would have banned the two highest-n perfect scorers on the highest-headroom task
    in the bank.
  - "Add no framing, hedging or politeness." Refuted: +1.54pp in-finetune, +1.07pp out. Politeness's
    raw negative signal is entirely the length and clause structure it drags along in the adversarial
    tier.
  - "Never use a category word ('receptacle', 'container', 'item')." No content beyond rule 1. Among
    rewrites that keep every head noun, a category word is +1.05pp (26 cells, 25 tasks, 8 positive /
    8 negative). Among rewrites that *have* dropped a head noun, replacing it with a category word
    rather than another concrete name is +0.36pp (61 cells, 50 tasks) — the three-way ordering
    verbatim > concrete synonym > category word does not reproduce. The whole apparent effect is the
    dropped noun.
  - "Never restate the object by its purpose or with a relative clause." Dead in every specification:
    within the adversarial tier -0.40pp with 30.8% of tasks positive and 30.8% negative, +0.01pp
    natural, positive in oracle_board, regression t between -1.1 and +1.1. Its raw signal is pure
    tier composition — in-finetune, clause markers appear in 18.0% of adversarial episodes and in
    0% of natural, oracle_board and canonical ones, so every in-finetune clause-bearing episode is
    an adversarial-prompt episode.
  - "Do not add a generic surface or workspace noun ('on the table', 'in your workspace')." Refuted,
    and it splits by which word you pick rather than by the ontology. Within (task x tier),
    out-of-finetune, introducing a word the instruction lacked: "table" +3.43pp (8 cells, 19 rows),
    "area" +2.05pp (12 cells), "desk" +1.66pp (3 cells, 20 rows), "surface" -0.34pp (33 cells,
    53 rows, 11 positive), "workspace" -1.60pp (12 cells), "currently before you" -9.79pp (8 cells,
    12 rows, 11 of them adversarial). Only the last is clearly bad and it is a 12-row
    adversarial-tier idiom, not a rule about surfaces. The prohibition as posed would forbid
    "from the table", the lexical signature of several of the largest searched wins in the bank.
  - "Do not name a second object the instruction did not name." Null. Within (task x tier),
    out-of-finetune: -0.23pp (36 cells, 29 tasks, 11 negative). Its raw in-finetune signal of
    -20.46pp is head-noun loss wearing a different hat: restricted to rewrites that keep every head
    noun it is -3.44pp on 5 cells and 7 rows in-finetune, +0.15pp on 8 cells out.
  - "Ban commas and subordinate clauses." Refuted. A comma-or-relative-clause marker scores
    -2.67pp in-finetune within (task x tier) but on only 6 of 17 cells (worse than a coin flip),
    +1.39pp out-of-finetune (90 cells, 55 tasks, 27 negative), -2.66pp in-finetune among head-noun
    keepers (12 cells, 4 negative), and +4.34pp computed inside the in-finetune adversarial tier
    alone (10 cells). The residue is task fragility, not syntax: libero_goal:5 and :6 supply the
    worst cells and the effect disappears without them.
  - "Avoid unusual words." Proxy, not cause. Novel-word count is significant pooled (-0.64,
    t=-2.70) but collapses once head nouns are held fixed, and is collinear with length and tier.
  - "Add a spatial qualifier for disambiguation in cluttered scenes." Null once head nouns are held
    fixed (+0.48pp out, +0.36pp in). The apparent over-specification penalty in the raw data is the
    noun-replacement penalty in disguise.
  - "Expand 'put X on Y' into 'pick up X and place it on Y'." Null: +0.25pp in-finetune (10 cells,
    5 tasks), -1.04pp out-of-finetune (28 cells, 21 tasks).
  - "Freeze every preposition." Null (see rule 7); only "inside" survives.
  - "Prefer the destination noun over the source noun, or vice versa." No asymmetry available: on the
    58 tasks with two or more head nouns, keeping the first one verbatim is +7.41pp within
    (task x tier) (99 cells) and keeping the last one +6.86pp (77 cells).
    Rule 1 covers both, which is why it says "every".

OPEN QUESTIONS THIS BANK CANNOT ANSWER

  1. Whether the rulebook raises success out-of-distribution. It does not, measurably: its
     out-of-finetune effect is -0.01pp per task against passing the instruction through. It is
     downside insurance. Anyone reading it as a success-maximiser is reading it wrong.
  2. Where the headroom actually is. The 11 out-of-finetune tasks in the 20-70% canonical band hold
     36.5pp of mean headroom, and on those tasks rule-following is roughly neutral — the wins are
     idiosyncratic single phrasings, not a referential style. The rephraser cannot identify that band
     from instruction text, and the bank offers no feature that predicts it. Headroom is also
     search-depth confounded (the 20 out-of-finetune tasks with more than 25 phrasings tried show
     30.4pp mean headroom against 2.6pp for the 35 tasks with 25 or fewer), so `best_pct` is not
     achievable by one guess.
  3. Rule 3's true magnitude. The verb effect is 2-5pp with an interval spanning zero, measured on 10
     in-finetune tasks carrying only 4 distinct canonical verbs (open, push, put, turn). Whether
     verb-keeping matters for verbs the finetune set never saw is untested.
  4. Rule 6's accuracy requirement. It rests on one paired same-task example (red-rimmed 97.1% vs
     red-striped 20.0%). There is no scene ground truth in the bank, so the cost of a *false*
     descriptor cannot be estimated. Treat "only what the scene asserts" as a safety margin, not a
     measured constraint.
  5. Rule 4 on collision tasks that rules 1-3 cannot rephrase. libero_90:77 admits no
     rules-1-3-compliant rewrite in the bank at all, so on tightly-worded trained instructions the
     rulebook may leave the rephraser with nothing legal to say but the input. If that happens,
     rule 4 wins: reorder or add a scene descriptor before conceding a verbatim echo.
  6. Transfer of specific phrasings. 181 phrase strings were measured on more than one task; their
     across-task success correlation is 0.47 with mean absolute difference 26.7pp over 265 pairs,
     against 9.3pp over 162 repeat measurements of the same phrase on the same task. Wording
     effects are substantially scene-specific, which caps how strong any wording rule can be —
     including these.
  7. In-finetune coverage. All 10 in-finetune tasks come from one suite, none of their canonical
     strings contains a colour word, and the oracle_board tier within them was searched for high
     scores. Rule 1 is the only finding here robust enough to bet on unseen tasks; rules 2, 3 and 7
     are small effects measured mostly out-of-finetune, and rules 5 and 6 are permissions whose value
     is that they stop the rephraser from paying for constraints that buy nothing.
