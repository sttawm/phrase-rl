# The previous rule set (v1) and its observed failures

The file `../b4_phrasing_rules.md` (repo: results/analysis/b4_phrasing_rules.md)
was authored from files 01-07 only, BEFORE the rollout search in file 08
existed. It was then used to rewrite 12 unseen instructions. Verbatim failures
observed (no rollouts were run on these; failures are judged against file 08's
measured evidence):

1. Emitted "put the purple object on the black keyboard" — used the banned
   category word "object" when its trace failed to name the eggplant. v1 had
   no fallback for a failed trace mapping. (File 08: category words are
   catastrophic, −31pp and worse.)
2. Emitted "put the block on the dish" — chose "dish", the most toxic
   receptacle noun measured (−31 to −61pp), and "block" where measured
   preference is "cube" (+12 to +25pp).
3. Emitted "the teal-green cube" — compound/unusual color adjective, measured
   −30pp vs plain "green". v1's rule said take the trace's color word
   verbatim; the fix is to SIMPLIFY to a basic color word.
4. Emitted "soda can" for a coke can — v1 allowed the trace's generic name to
   override a brand token; file 08 measures the "coke" token at +14 to +31pp
   over soda/cola/bare-can.
5. v1 BANNED the verbs "set" and the word "please" — file 08 shows both are
   harmless (set ≈ put across 5+ boards; set WON a confirmed board).
6. v1 required "upright" for containment — file 08 shows orientation words
   are unnecessary (pooled ~+8-10pp for dropping).
7. v1's evidence for banning came partly from confounded pairs (e.g. "Set the
   spoon exactly in the middle of the towel" blamed "set" for the harm done
   by "exactly in the middle").

The v2 rule set should keep v1's load-bearing structure (trace-noun adoption,
description deletion, plain register, length budget) and repair these seven
failures, using file 08's measured pairs as the authority wherever it
conflicts with files 03-07.
