# Task: write phrasing rules for a robot instruction rewriter

## The situation

A robot manipulation policy executes short natural-language instructions
("put the spoon on the towel"). The policy is frozen — it cannot be
retrained. How well it performs depends heavily on the exact wording of the
instruction it is given: the same task phrased two different ways can differ
by 40+ percentage points of measured success.

At test time, instructions arrive in adversarially reworded forms (verbose,
indirect, with objects described rather than named). A rewriter model
converts each incoming instruction into the phrasing the policy executes
best. Alongside the incoming instruction, the rewriter receives a short
machine-written "trace": a one-or-two-sentence description of the scene plus
a mapping from the instruction's object descriptions to plain object names
(see file 07 for real examples).

## Your job

Study the files in this folder and author an explicit RULE SET the rewriter
should follow. The rules must be:

1. **Task-agnostic** — never mention a specific task or object from the
   evaluation data. Not "prefer cube over block" but "when several concrete
   nouns fit the traced object, prefer the one more frequent in the training
   corpus for that object class."
2. **Executable** — each rule must be applicable given ONLY: the incoming
   instruction, the trace, and the training corpus (files 01/02). No rule
   may require running the robot or knowing measured success rates.
3. **Evidence-cited** — each rule names the file(s) and case(s) that
   motivate it.

## The files

- `01_bridge_gt_instructions.txt` — The complete list of 17,297 unique
  instructions from the robot policy's training data, one per line. This is
  the phrasing distribution the policy was trained to follow. No labels.
- `02_oxe_paraphrases_SAMPLE60keys.csv` — A sample of 60 instruction
  "clusters" from the policy's finetuning augmentation: each row pairs a
  training instruction (gt) with one of its ~38 benign paraphrases that the
  policy also saw during training. 235,764 such paraphrases exist in total.
- `03_contrast_pairs_139.json` — 139 statistically confirmed preference
  facts. Each record is two phrasings of the SAME task, both executed many
  times in simulation; "better" beat "worse" by at least 6 percentage points
  of rollout success with z>1.96 under conservative (layout-clustered)
  errors, each phrase measured over ≥200 episodes. These are the
  highest-confidence facts about which phrasings the policy executes better.
- `04_greedy_cells_all_arms.csv` — A chart of rollout successes for 8
  simulated tasks. Each task is identified by its nominal instruction; each
  row is one alternative phrasing of that task with its measured success
  rate over at least 288 rollouts (up to 1,152 where measurements were
  pooled). 73 phrasings total. Compare phrasings within a task to see what
  wording the policy executes best.
- `05_sampled_pools_per_draw.csv` — Individual sampled phrasings (8 random
  draws per task from various generator models) with their measured success
  over 24-96 rollouts each; 435 rows. The best draw in each pool is flagged
  (is_pool_oracle). CAUTION: per-row success is noisy (SE ~10pp at n=24)
  and pool winners are selected on noise — use these rows for the STRUCTURE
  of what wins, never for exact rankings.
- `06_mechanism_cases.md` — Curated case studies with measured numbers:
  side-by-side phrasings of the same task showing large success differences
  and the apparent cause. Each case includes the task's nominal instruction
  and its measured rate.
- `07_selftraces_val8.md` — Real examples of the trace the rewriter receives
  at test time: a short model-written scene description plus a mapping from
  the instruction's object descriptions to plain object names. Your rules
  must be executable given only the instruction, a trace like these, and the
  training corpus.

## How to weigh the evidence

CI-cleared pairs (03) > large-n success cells (04) > pooled patterns across
many noisy draws (05) > any single high-scoring draw. Treat oracle-flagged
draws in 05 as inspiration for phrasing STRUCTURE, never as proof that the
specific phrase is best.

## Output

A numbered rule set (aim for the smallest set that covers the evidence),
each rule with: the rule itself, when it applies, one worked example on a
HYPOTHETICAL instruction (invent your own — do not reuse evaluation tasks),
and its evidence citations. Follow with a short section on known tensions
(cases where rules conflict and how to arbitrate, e.g. Case 2 vs Case 3 in
file 06).
