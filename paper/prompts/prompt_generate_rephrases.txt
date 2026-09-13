You are producing test instructions for a robot manipulation task, to measure how
sensitive a policy is to the way a request is worded.

THE TASK'S OWN INSTRUCTION (the wording the policy was trained on):
{{instruction}}

Write two sets of rewordings that keep the goal EXACTLY the same. The object, the
destination, and the action must be identical in every line — only the wording
changes. A line that changes which object is moved, or where it goes, is useless
to us and worse than no line at all.

## NATURAL ({{n_natural}} lines)
What a person would actually say to a robot in a kitchen or on a workbench:
fluent, unremarkable, varied in structure. Vary the verb, the article, the word
order and the level of detail across the set — not {{n_natural}} versions of one
sentence. Some may name the object by colour or material if that is unambiguous.
These should be EASY.

## ADVERSARIAL ({{n_adversarial}} lines)
Awkward, ornate, or indirect phrasings that a policy is likely to handle badly
while a person would still understand them: heavy subordinate clauses, indirect
reference ("the thing you would drink when thirsty"), unusual register, polite
circumlocution, or front-loaded conditions. Still unambiguous to a human, and
still the SAME goal. These should be HARD.

Reply with exactly this, and nothing else — no commentary, no numbering, no
markdown fences:

NATURAL
<one instruction per line>

ADVERSARIAL
<one instruction per line>
