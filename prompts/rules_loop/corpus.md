You are building a vocabulary reference for the instruction corpus a robot policy
was trained on. A later step uses this file to decide, for any given word,
**whether the policy has ever seen it** — so the file must be a complete lookup
table, not a summary. A summary is useless for that job: the question is always
"is THIS word in the corpus", and a word absent from your file will be treated as
absent from the corpus.

Read the files carefully and count exactly. Where a count is genuinely
approximate, say so rather than presenting an estimate as exact.

## Inputs

Read everything under {{inputs_dir}} (start by listing it and reading any README).
The raw instruction list is the important one; the other files are context.

## What to compute

Tokenize the instructions (lowercase, strip punctuation, split on whitespace).
Count every token. Then write **the entire vocabulary above a frequency floor of
5 occurrences** — words occurring fewer than 5 times are excluded as noise, and
say so in the file. Do not truncate, sample, or abbreviate the list for length.
Completeness is the entire point of the artifact.

## Output format

Write to {{out_file}}, in this structure:

```
# Corpus vocabulary — exact membership test

<one line: how many instructions, how many distinct after the floor, what the
floor was, and how tokenization was done>

A word is corpus-present if and only if it appears below. Presence licenses a
token; it never makes a token good — frequency is not outcome.

**Common (n>=50):** word 1234, word 987, word 654, ...
   (every word at or above 50, with its exact count, descending by count)

**Present (n=5-49):** word, word, word, ...
   (every remaining word above the floor, alphabetical, no counts needed)

## Shape of the corpus
<typical instruction length; the usual sentence form; how instructions begin;
whether articles/determiners are usually present — with counts or percentages>

## Register
<how objects get described: by category, colour, material, brand, function.
Give the counts that support each claim.>

## Notable absences
<word classes or constructions conspicuously rare or missing given the domain.
Name specific words you checked for and did not find, or found below the floor.>
```

Keep both vocabulary tiers on single lines, comma-separated, exactly as shown —
the consumer greps them.

Describe only what the corpus contains. Do not speculate about what the policy
prefers, and do not recommend phrasings; later steps decide that from measured
outcomes, and a prior smuggled in here would contaminate them.

Reply with a one-line confirmation and the two tier sizes once the file is written.
