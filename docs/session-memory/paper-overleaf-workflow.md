---
name: paper-overleaf-workflow
description: "paper edits go through the Overleaf git clone at ../phrase-rl-paper (pull before, push after); phrase-rl/paper/main.tex is a mirror, not the canonical"
metadata: 
  node_type: memory
  type: project
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-08-10T16:51:41.261Z
---

The paper ("Improving VLA Performance by Rephrasing Task Instructions", IEEE
conf format) is edited via Overleaf git integration, set up 2026-08-10:

- **Canonical**: the Overleaf project, cloned at `/Users/sttawm/dev/robotics/phrase-rl-paper`
  (remote `https://git.overleaf.com/6a7269aaf9f91194e71e1092`, branch `main`).
- **Auth**: token in `~/.zshrc` as `OVERLEAF_GIT_TOKEN`, wired into a host-scoped
  credential store at `~/.overleaf-git-credentials` (chmod 600) via
  `credential.https://git.overleaf.com.helper` — never put the token in a remote
  URL or a `ps`-visible command.
- **Loop**: `git pull` before ANY edit (the user edits live in Overleaf), push
  after each section, then `cp main.tex ../phrase-rl/paper/main.tex` and commit
  the mirror. The mirror compiles locally because of
  `\graphicspath{{./}{../results/charts/}}`; figures live at Overleaf project
  root and in `results/charts/` (two figures differ between the copies:
  reward_grids_gripper.png, system_overview.png — Overleaf's are canonical).
- The user prompts one section at a time; do not rewrite unprompted sections.

**Why:** the user's Overleaf copy drifts ahead of any pasted snapshot (it
already carried results-section notes the paste lacked), so editing the repo
mirror without pulling first silently discards their edits.

**How to apply:** before every paper edit: `cd ../phrase-rl-paper && git pull`.
Unverified numbers flagged 2026-08-10: the abstract's "+25% adversarial / +30%
natural" could not be reproduced from the closing table (naturals 28.45 →
gemini rules 33.92 = +19.2% rel; repair fractions 72% gemini / 36% claude) —
ask the user for the source before propagating those numbers into new sections.
Related: [[phrase-search-doc-sync]].
