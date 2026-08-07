You are assembling a reference document about the instruction corpus a robot
policy was trained on. A later step will use this document to reason about which
phrasings that policy is likely to understand.

Read the corpus files under {{inputs_dir}} (use Read/Grep/Glob freely — start by
listing the directory and the README if one exists).

Write your findings to {{out_file}} as markdown. Cover at least:
- Size: how many instructions, how many distinct.
- Vocabulary: the most frequent nouns, verbs, and modifiers, with counts. Which
  object words are common, which are rare, which never appear.
- Grammatical shape: typical length, typical sentence form, how instructions
  usually begin, whether articles/determiners are usually present.
- Register: how instructions describe objects — by category, by color, by
  material, by brand, by function.
- Notable absences: constructions or word classes that are conspicuously rare.

Be concrete and quantitative — cite counts and give verbatim examples. Do not
speculate about what the policy "prefers"; describe only what the corpus contains.

Reply with a one-line confirmation once the file is written.
