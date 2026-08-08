You are turning proposed experiments into concrete phrases to measure, for a
rulebook that rewrites instructions given to a robot policy.

## Working files

  {{summary_file}}
      One row per task — the complete set of tasks you may probe. For each: how
      many phrases have been measured on it, the best, worst and median score
      (normalised WITHIN each task, so spread is comparable across tasks but
      the levels are not),
      the SPREAD between best and worst, how many of its phrases carry a real
      rollout number rather than an estimate, and how many rest on thin sampling.
      Sorted by spread, descending. **Every task you name must appear here**,
      spelled exactly as it is spelled here; a probe naming anything else is
      discarded.

  {{evidence_file}}
      Every phrase measured so far, task by task. Use it to see what has already
      been tried on a task before proposing something for it — a phrase that
      merely restates an existing measurement buys nothing.

## Proposed experiments

From the rulebook audit. May be empty, in which case propose nothing and reply
with no lines at all:

{{suggestions}}

## Choosing where to spend the probes

You decide which tasks to probe. Let the suggestions lead where they name
something specific, and otherwise spend the budget where measurement would
actually change what we know:

- tasks with few phrases measured, where almost anything is new information;
- tasks whose phrases are bunched at similar scores, where the effect of wording
  is currently invisible and may simply be unmeasured;
- tasks with a large spread but thin sampling behind it, where the gap could be
  real or could be noise;
- a mix of tasks rather than every probe on one, unless the suggestions genuinely
  point at one task.

A task that is already densely measured and clearly separated buys the least.

## Writing the phrases

Each phrase must be a plausible instruction for its task — a real thing someone
might say — not a nonsense string. Vary one thing at a time where you can, so the
resulting measurement is interpretable: two phrases differing in a single word
tell you what that word does, while two differing in five tell you nothing in
particular.

You may also re-list a phrase that is ALREADY in the evidence. Doing so measures
it again on fresh contexts and the two measurements combine, so a phrase with a
low `n_ctx` whose result matters becomes more precisely known. Use this when a
difference you want to reason about is too close to call at its current sample
size, rather than treating a thin measurement as settled.

## Format

Reply with one phrase per line, in EXACTLY this format and nothing else:

[task name] the phrase to measure

Maximum {{max_probes}} lines.
