export const meta = {
  name: 'libero-distill-draw1',
  description: 'Rulebook draw 1 for pi0.5/LIBERO: lens analysts over the single-edit evidence, adversarial refuters per candidate rule, one synthesizer under the frozen distill prompt',
  phases: [{ title: 'Analyze' }, { title: 'Refute' }, { title: 'Synthesize' }, { title: 'Audit' }],
}
const REPO = '/Users/tararaffi/dev/robotics/phrase-rl'
const EVID = `${REPO}/results/analysis/pi05_bank/distill_evidence_v2.csv`
const PROMPT = `${REPO}/results/analysis/pi05_bank/distill_prompt_v2_filled.md`
const OUT = `${REPO}/results/analysis/pi05_bank/rulebooks/v2_draw1.md`
const GROUND = `Evidence: ${EVID} — 271 instruction phrasings on 58 LIBERO tasks for a frozen pi0.5 robot policy, columns task, category, phrase, success (% of 50 rollouts on identical initial states), n. Rows sharing a task are directly comparable; "canonical" is the string the policy was fine-tuned on for that task; every other row differs from some sibling row by exactly one edit of the named category. Differences under ~18 points are within measurement noise at n=50. You may read the file with any tool (python/pandas via ${REPO}/.venv/bin/python is available). Do not modify any file unless told to. You know nothing about which tasks are held out; never guess at them.`

const LENSES = [
  { key: 'nouns', focus: 'object and place nouns: renamings, hypernyms, synonyms, rare vs trained names, dropping or adding a noun modifier' },
  { key: 'verbs', focus: 'verbs and sentence frames: verb substitutions, particles, imperative vs declarative/question/desire/passive forms, clause structure' },
  { key: 'modifiers', focus: 'colour words and modifiers, words added or removed, determiners, capitalisation and punctuation' },
  { key: 'preps', focus: 'prepositions and particles, spatial relation wording, locator phrases' },
  { key: 'consistency', focus: 'cross-task consistency: which effects replicate across two or more tasks versus effects that appear in one task only; which categories are reliably harmless' },
]
const RULES_SCHEMA = { type: 'object', properties: { rules: { type: 'array', items: { type: 'object', properties: {
  rule: { type: 'string' }, support: { type: 'array', items: { type: 'string' } }, counterexamples: { type: 'array', items: { type: 'string' } } },
  required: ['rule', 'support', 'counterexamples'] } } }, required: ['rules'] }
const VERDICT = { type: 'object', properties: { refuted: { type: 'boolean' }, reason: { type: 'string' } }, required: ['refuted', 'reason'] }
const BOOK_SCHEMA = { type: 'object', properties: { rules_text: { type: 'string' }, n_rules: { type: 'number' } }, required: ['rules_text', 'n_rules'] }
const AUDIT = { type: 'object', properties: { ok: { type: 'boolean' }, issues: { type: 'array', items: { type: 'string' } } }, required: ['ok', 'issues'] }

const REFUTERS = [
  r => `Counterexample hunter. Candidate rule: "${r.rule}". Claimed support: ${r.support.join(' | ')}. Search the evidence for cases that contradict the rule's direction (a phrase the rule would produce scoring lower than the phrase it would replace, by more than noise), or where the rule's premise is not what the data shows. If the claimed support rows do not exist or do not say what is claimed, that is a refutation. Default to refuted=true if uncertain.`,
  r => `Overfit and generality checker. Candidate rule: "${r.rule}". Support: ${r.support.join(' | ')}. Is the effect present on at least two different tasks (not one task, not one ladder on one task)? Is the rule stated so a rewriter could apply it to instructions, objects and scenes not in the evidence, rather than a lookup table of specific words from single tasks? A rule that is really a one-task observation, or that only restates a task-specific noun, is refuted. Default to refuted=true if uncertain.`,
  r => `Harm and ambiguity checker. Candidate rule: "${r.rule}". Support: ${r.support.join(' | ')}. Would applying this rule mechanically to a person's ordinary instruction risk changing its meaning, introducing a referent ambiguity in a cluttered scene (several bottles, several bowls), or contradicting another strongly supported pattern in the same evidence (e.g. renaming an object the policy was trained on)? Refute if the expected harm outweighs the measured benefit or if the rule needs information a rewriter would not have. Default to refuted=true if uncertain.`,
]

const surviving = await pipeline(LENSES,
  l => agent(`${GROUND}\n\nYou are one of five analysts. Your lens: ${l.focus}. Read the whole evidence file, then propose at most 6 candidate rewriting rules for a rulebook whose purpose is: given any incoming instruction for this robot, rewrite it so the robot is most likely to succeed. Each rule must be grounded in consistent score differences across tasks, not single examples, and must be general enough to apply to instructions and objects not in the evidence. For each rule give the supporting rows verbatim as "task | phrase | success" strings (at least 2 tasks where possible) and list honest counterexamples from the data. Rules about what NOT to change (protecting wording that works) are as valuable as rules about what to change.`,
    { label: `analyze:${l.key}`, phase: 'Analyze', schema: RULES_SCHEMA, effort: 'high', model: 'opus' }),
  (res, l) => parallel((res ? res.rules : []).map(r => () =>
    parallel(REFUTERS.map((mk, i) => () => agent(`${GROUND}\n\n${mk(r)}`, { label: `refute${i + 1}:${l.key}`, phase: 'Refute', schema: VERDICT, effort: 'high', model: 'opus' })))
      .then(vs => ({ ...r, lens: l.key, refutations: vs.filter(Boolean).filter(v => v.refuted).map(v => v.reason), survives: vs.filter(Boolean).filter(v => v.refuted).length < 2 }))))
)
const cands = surviving.filter(Boolean).flat().filter(Boolean)
const kept = cands.filter(c => c.survives)
log(`${cands.length} candidate rules, ${kept.length} survive adversarial review`)

const notes = kept.map((c, i) => `${i + 1}. [${c.lens}] ${c.rule}\n   support: ${c.support.join(' | ')}\n   counterexamples: ${c.counterexamples.join(' | ') || 'none listed'}${c.refutations.length ? `\n   one refuter objected: ${c.refutations.join(' / ')}` : ''}`).join('\n')
const synthPrompt = (extra) => `${GROUND}\n\nYour task is the file ${PROMPT}: read it in full — it contains the exact distillation instructions (the paragraph before and after the evidence table) and the same evidence as a table. Follow those instructions literally: write a rulebook of 10-20 numbered rules for rewriting any incoming task instruction so that the robot is most likely to succeed, grounded in consistent score differences across tasks, general enough for unseen instructions and objects.\n\nBelow are candidate rules from five independent analysts; each survived three adversarial reviewers who checked it against the evidence (objections that were raised are noted). Use them as verified notes, merge duplicates, drop anything you cannot ground yourself, and add rules the analysts missed if the evidence supports them.\n\n${notes}\n${extra}\nWrite the result to ${OUT} (create the directory if needed) in exactly this format: a line "===RULES===", then the numbered rules (one per line, imperative, self-contained — the rules are applied mechanically by another model that sees only the rules, the incoming instruction and a scene description, and cannot see this evidence), then a line "===RATIONALE===", then for each rule number the evidence rows that ground it. Return the RULES section text and the rule count.`
let book = await agent(synthPrompt(''), { label: 'synthesize', phase: 'Synthesize', schema: BOOK_SCHEMA, effort: 'xhigh', model: 'opus' })
const audit = await parallel([
  () => agent(`${GROUND}\n\nAudit the rulebook at ${OUT} against the evidence. For every rule: is it grounded in at least two tasks in the evidence (name them), does any evidence row contradict it by more than noise, is it stated so it can be applied mechanically to unseen instructions, and does it contradict another rule in the book? Also check the file format: "===RULES===" header, 10-20 numbered rules, "===RATIONALE===" section. Report only concrete issues, quoting the rule number.`, { label: 'audit:grounding', phase: 'Audit', schema: AUDIT, effort: 'high', model: 'opus' }),
  () => agent(`${GROUND}\n\nAudit the rulebook at ${OUT} as the model that must APPLY it: you will receive only these rules, an incoming instruction and a scene description. Is every rule unambiguous and executable without seeing the evidence? Does any rule require knowing the policy's training vocabulary or a specific task? Could two rules fire on the same instruction and conflict? Would any rule change what object is moved or where? Report only concrete issues, quoting the rule number.`, { label: 'audit:applier', phase: 'Audit', schema: AUDIT, effort: 'high', model: 'opus' }),
])
const issues = audit.filter(Boolean).flatMap(a => a.ok ? [] : a.issues)
if (issues.length) {
  log(`${issues.length} audit issues -> one revision pass`)
  book = await agent(synthPrompt(`\nA previous draft of the book is already at ${OUT}. Two auditors found these issues; revise the file to fix each one (or leave a rule unchanged only if the objection is wrong on the evidence, and say so in the RATIONALE):\n- ${issues.join('\n- ')}\n`), { label: 'revise', phase: 'Synthesize', schema: BOOK_SCHEMA, effort: 'xhigh', model: 'opus' })
}
return { candidates: cands.length, kept: kept.length, audit_issues: issues, n_rules: book ? book.n_rules : null, rules: book ? book.rules_text : null }