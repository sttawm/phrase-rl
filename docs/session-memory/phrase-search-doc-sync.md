---
name: phrase-search-doc-sync
description: Keep PHRASE-SEARCH.md Method/Purification sections in sync whenever the measurement protocol changes (user order 2026-07-23)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-07-23T00:08:04.013Z
---

User order (2026-07-23): "make sure you update the Method / Purification rounds in PHRASE-SEARCH.md when it's relevant."

**Why:** PHRASE-SEARCH.md is the project's evidence-of-record for phrasing effects; its Method header is what makes every number interpretable. Protocol changes (layout coverage, reps, stratified pooling, board naming like `xcert_pure_*`) silently invalidate the header if not synced. The user reads this file directly and has caught staleness/inconsistency repeatedly (split-mixing verdicts, pure1 wording).

**How to apply:** any time a board protocol, pooling rule, naming convention, or SE tier changes, update the Method + Purification-rounds paragraphs in the same commit that introduces the change. Also keep verdict lines derived from the pooling rule currently described in Method (layout-stratified, uniform reps per layout — see [[checkpoint-archive-discipline]] era, 2026-07-22 stratified rewrite). Plain english, no jargon: the user pushes back on compressed stats-speak.
