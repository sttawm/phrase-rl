export const meta = {
  name: 'v2-distill-replicates',
  description: 'Replicate rulebook draws (2 per evidence diet) to estimate between-draw variance; opus swarm',
  phases: [
    { title: 'Distill', detail: '3 arms x 3 lens-distillers' },
    { title: 'Critique', detail: 'one adversarial critic per arm' },
    { title: 'Synthesize', detail: 'one synthesizer per arm' },
  ],
}

const REPO = '/Users/sttawm/dev/robotics/phrase-rl'
const BOOK = {
  type: 'object',
  properties: {
    book: { type: 'string', description: 'The complete rulebook: ===RULES=== numbered rules, then ===RATIONALE=== short evidence notes' },
    key_evidence: { type: 'string', description: '3-6 bullet lines: the measurements that drove the biggest rules' },
  },
  required: ['book'],
}
const CRIT = {
  type: 'object',
  properties: {
    verdicts: { type: 'string', description: 'Per-candidate: strongest rules, broken rules (with the evidence they violate), leakage findings, overfit findings' },
    must_fix: { type: 'string', description: 'Numbered list of mandatory fixes for the synthesizer' },
  },
  required: ['verdicts', 'must_fix'],
}

const APPLYCTX = `The book will be applied MECHANICALLY, one instruction at a time, by three
different rule-following models (Claude, Gemini, a 9B Qwen) that see only your rules, the
incoming instruction, and a scene-description trace. Rules must be executable by the weakest
of them: crisp triggers, no unstated judgment. Rule 1 must pin the output format (one line,
the rewritten instruction, nothing else, never any reasoning text). The target is a frozen
pi0 policy: phrasing choices move task success by tens of points.`

const ARMS = [
  {
    key: 'train_only',
    brief: `EVIDENCE DIET (train-only): ${REPO}/results/rules_runs/v2distill/train_only/ contains
evidence.csv (+README, summary): 4,544 phrases over 220 tasks scored ONLY by the gripper-reward
proxy (columns z, grip, logit, score; NO rollout ground truth anywhere), plus corpus_stats.md
(training-corpus text statistics), 08_phrase_search.md and 06_mechanism_cases.md (measured field
notes, val8 sections removed). CONSTRAINT: the val8/simulation tasks and the sealed suite are
DELIBERATELY absent from your evidence; do not speculate about them, and write rules grounded in
corpus properties (vocabulary membership, register, form) that generalize to any task drawn from
this corpus. The proxy ranks phrases within a task; treat its levels as ranking signal, never as
success rates.`,
  },
  {
    key: 'sim_only',
    brief: `EVIDENCE DIET (sim-only): ${REPO}/results/rules_runs/v2distill/sim_only/ contains
evidence.csv (+README, summary): 516 phrases over 8 tasks, every row a REAL rollout success rate
(gt_success 0-100, n_ctx = episodes; weight by n_ctx). There are NO proxy rows, NO corpus
statistics, NO training-data artifacts: treat the policy as a BLACK-BOX VLA you know only through
these measured rollouts. Be specific where the data is deep (high-n contrasts) but do not overfit:
a noun mapping seen on one task is a hypothesis, not a rule -- state the principle it instantiates
so the applier can extrapolate, and prefer rules supported by contrasts on several tasks.`,
  },
  {
    key: 'both',
    brief: `EVIDENCE DIET (both): ${REPO}/results/rules_runs/v2distill/both/ contains the FULL bank
evidence.csv (+README, summary): 5,236 phrases over 228 tasks -- 516 rows carry REAL rollout
success (gt_success, weight by n_ctx; trust over proxy on conflict), the rest are gripper-proxy
ranking rows -- plus corpus_stats.md, 08_phrase_search.md, 06_mechanism_cases.md,
09_rules_v1_failures.md, 03_contrast_pairs_139.json. Your book should COMBINE the registers: the
corpus-level principles the proxy evidence supports at breadth, sharpened by what real rollouts
proved and corrected where documented failures show past rules misfired.`,
  },
]

const LENSES = [
  ['register', 'Lead with input triage: classify the incoming instruction (plain vs ornate vs telegraphic vs frozen-form) and decide whether to edit AT ALL before any rewrite rule. Mine the evidence for which input registers were unbeatable vs repairable.'],
  ['mechanism', 'Lead with measured contrasts: build every rule from the largest reliable phrase-pair effects in the evidence (noun choice, added/removed detail, verb frames), each rule carrying the contrast that justifies it.'],
  ['conservative', 'Lead with harm avoidance: identify every edit class the evidence shows can backfire and write the tightest whitelist of provably-safe repairs; when in doubt the book must say copy unchanged.'],
]

phase('Distill')
const REPS = [2, 3]
const candidates = await parallel(ARMS.flatMap(arm =>
  REPS.flatMap(rep =>
  (rep === 2 ? LENSES : [LENSES[2], LENSES[0], LENSES[1]]).map(([lens, lensBrief]) => () =>
    agent(
      `You are distilling a phrasing rulebook for a frozen pi0 robot manipulation policy.\n\n${APPLYCTX}\n\n${arm.brief}\n\nLENS -- ${lens}: ${lensBrief}\n\nThis is an INDEPENDENT draw: do not attempt to reproduce any canonical or previous answer. Read the evidence yourself and reach your own conclusions.\n\nRead the files (start with evidence.csv.README.md, then evidence_summary.csv, then dig into evidence.csv and the other artifacts). Then write the best complete rulebook you can under your lens: 8-18 numbered rules, each executable and evidence-grounded. Return it via structured output.`,
      { label: `distill:${arm.key}:r${rep}:${lens}`, phase: 'Distill', schema: BOOK, effort: 'high', model: 'opus' })
  ))))

phase('Critique')
const GROUPS = []
ARMS.forEach((arm, ai) => REPS.forEach((rep, ri) => {
  const off = (ai * REPS.length + ri) * 3
  GROUPS.push({ arm, rep, books: candidates.slice(off, off + 3).filter(Boolean) })
}))
const critiques = await parallel(GROUPS.map(G => () => {
  const arm = G.arm
  const books = G.books.map((c, i) => `--- CANDIDATE ${i + 1} ---\n${c.book}`).join('\n\n')
  const leakClause = arm.key === 'train_only'
    ? 'LEAKAGE CHECK (mandatory): the evidence excluded the val8/sim tasks (carrot/eggplant/spoon/cube/coke-can micro-suite) and the sealed suite. Flag ANY rule or example that names objects or tasks the evidence cannot support.'
    : arm.key === 'sim_only'
      ? 'OVERFIT CHECK (mandatory): the evidence covers only 8 tasks. Flag every instance-level mapping stated as a bare rule without its generalizing principle, and every rule a novel-object task would break.'
      : 'CONSISTENCY CHECK (mandatory): flag rules where the proxy evidence and rollout evidence disagree and the book picked the proxy side.'
  return agent(
    `Three candidate phrasing rulebooks for the same frozen pi0 policy are below. The evidence diet they were distilled from lives in ${REPO}/results/rules_runs/v2distill/${arm.key}/ -- consult it to check claims.\n\n${leakClause}\n\nAlso attack: rules a mechanical applier (including a 9B model) cannot execute; rules contradicting the measured evidence; missing rules the evidence clearly supports.\n\n${books}\n\nReturn per-candidate verdicts and a numbered must-fix list.`,
    { label: `critique:${arm.key}:r${G.rep}`, phase: 'Critique', schema: CRIT, effort: 'high', model: 'opus' })
}))

phase('Synthesize')
const finals = await parallel(GROUPS.map((G, ai) => () => {
  const arm = G.arm
  const books = G.books.map((c, i) => `--- CANDIDATE ${i + 1} ---\n${c.book}`).join('\n\n')
  const crit = critiques[ai]
  return agent(
    `Synthesize ONE final phrasing rulebook for the ${arm.key} evidence diet (files in ${REPO}/results/rules_runs/v2distill/${arm.key}/ -- consult them to resolve disputes with measurements, not taste).\n\n${APPLYCTX}\n\n${arm.brief}\n\nTHE THREE CANDIDATES:\n${books}\n\nTHE CRITIC'S VERDICTS:\n${crit ? crit.verdicts : '(critic unavailable)'}\n\nMANDATORY FIXES:\n${crit ? crit.must_fix : '(none)'}\n\nTake the strongest structure as the spine, graft the best rules from the others, apply every mandatory fix, and keep it executable by the weakest applier. 10-18 numbered rules. Return via structured output.`,
    { label: `synthesize:${arm.key}:r${G.rep}`, phase: 'Synthesize', schema: BOOK, effort: 'xhigh', model: 'opus' })
}))

const out = {}
GROUPS.forEach((G, i) => { out[`${G.arm.key}_r${G.rep}`] = finals[i] ? finals[i].book : null })
return out