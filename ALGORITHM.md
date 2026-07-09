
vlm_1 = gemini
vlm_2 = qwen


// Pre-Compute:
//
// Get reasoning traces from Gemini
// 
for ep in training_episodes:
  task = ep.task // TODO: Should this be the Red-Team phrase, to match test time?
  obs = ep.observation
  trace = gemini.prompt(get_cover_reasoning_trace_prompt(task, obs))


// Training
// 
for (ep, trace):

  batch = []

  // Prompt Qwen for rephrases (VLM: image goes in the prompt too)
  //  
  rephrases = qwen.prompt(get_phrases_prompt(n=16, ep.image, ep.task, trace))

  //  Remove phrases that drift too far from the original
  //  (current impl: frozen BASE qwen as the judge — free, in-process;
  //   gemini is a one-flag switch, ~14 short calls/step, <$5/run)
  // 
  rephrases = rephrases.filter(is_good, gemini)
  
  // Compute rewards with the FROZEN pi0 VLA on ground-truth actions 
  gt = ep.ground_truth_actions
  phrases_with_rewards = [(p, compute_reward(pi0, p, gt)) for p in rephrases] // VARY: L2 vs Flow-loss
  phrases_with_advantages = to_advantages(rewards)  // z-normalize within the episode's group
  phrases_with_positive_advantages = phrases_with_advantages.filter(lambda (_, advantage): advantage > 0)

   
  for (phrase, advantage) in phrases_with_positive_advantages:
    batch.append((trace, phrase, advantage))

  if batch >= batch_size:
    gradients = []
    for (trace, phrase, advantage):
      context = ep.image + trace + single_phrase_prompt
      // KL leash is NOT optional: v1 collapsed without it (KL->2.0, degenerate output).
      // loss = -advantage * logprob(phrase | context) + beta * KL(qwen || qwen_base), beta=0.15
      gradients += qwen.forward_with_weight(in=context, expected_out=phrase, weight=advantage, kl_beta=0.15)
    qwen.update(gradients)
    batch = []

// Testing
// (trace comes from GEMINI at test time too — matches training exactly; decided 2026-07-10)
// Run BOTH modes: (a) nominal instructions (our benchmark), (b) red-team instructions
// (CoVer's Table-3 protocol: their 4 ERT phrases, reset seeds 1000-1049, 150-step horizon).
// 
success = 0
for (task, observation, goal_condition) in test_data:
  red_team_phrase = gemini.prompt(get_cover_read_teamed_phrase(task, observation)) // ERT is image-conditioned (their initial-state frame)
  trace = gemini.prompt(get_cover_reasoning_trace_prompt(red_team_phrase, observation))
  optimized_task_phrase = qwen.prompt(observation.image + trace + single_phrase_prompt)
  // success = EXECUTE the phrase in SIMPLER and read env success — never a predicate on the text
  if env.rollout(pi0, optimized_task_phrase, observation).success:
    success += 1
