You are turning proposed experiments into concrete phrases to measure.

Read the working file for what has already been measured, so you do not propose
phrases that are already in the bank:

  {{evidence_file}}

PROPOSED EXPERIMENTS (from the rulebook audit; may be empty, in which case
propose nothing and reply with no lines):
{{suggestions}}

Choose which tasks to probe yourself, from the evidence file. Use ONLY task names
that appear there, verbatim — that file is the permitted set, and a probe naming
anything else is discarded.

Prefer tasks where more measurement would actually change what we know: ones with
few phrases, ones whose phrases are bunched at similar scores, or ones where the
audit's questions bite. Spending all the probes on a task that is already
densely measured buys little.

Each phrase must be a plausible instruction for its task — a real thing someone
might say — not a nonsense string. Vary one thing at a time where you can, so
the resulting measurement is interpretable.

You may also re-list a phrase that is ALREADY in the evidence. Doing so measures
it again on fresh contexts, and the two measurements combine — so a phrase with a
low `n_ctx` whose result matters becomes more precisely known. Use this when a
difference you want to reason about is too close to call at its current sample
size, rather than treating a thin measurement as settled.

Reply with one phrase per line in EXACTLY this format, and nothing else:
[task name] the phrase to measure

Maximum {{max_probes}} lines.
