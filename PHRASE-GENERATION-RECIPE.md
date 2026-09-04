# Phrase-set and trace generation — the recipe, as actually used

Everything below is what the bridge/π0 work ran, with file paths and the
image-vs-text answer for each stage. π0.5/LIBERO ports already exist where noted.

---

## 1. Bank phrases (natural + adversarial), for TRAINING-side evidence

**Prompt:** `prompts/rules_loop/generate.md` — one prompt, two blocks
(`## NATURAL {{n_natural}} lines` and `## ADVERSARIAL {{n_adversarial}} lines`),
asked for in a single response so the natural block stays anchored as the easy
end of a contrast.

**Runner:** `scripts/generate_phrases.py`
**Model:** `gemini-pro-latest`, temperature 1.0
**Image:** **NO — text-only.** The generator sees only the task's own
instruction string.
**Scale used (bridge):** 7 natural + 2 adversarial per task, 228 tasks →
1,888 naturals + 991 adversarials.

Two features of this prompt matter more than the model choice:
- it **names the variation axes** ("vary the verb, the article, the word order
  and the level of detail across the set — not N versions of one sentence");
- on top-up calls it appends **every phrase already written for that task**
  under `ALREADY WRITTEN for this task`, so later calls cannot re-cover ground.

Dropping either collapses the register (measured: 18 distinct sentence-initial
words without them vs 137 with, on the same model).

**π0.5 port, already written:** `scripts/gen_pi05_bank_phrases.py` — same
`generate.md` prompt, text-only, 10 natural + 10 adversarial per train task
from `results/analysis/pi05_bank/splits.json`.

---

## 2. Sealed EVALUATION adversarial set (the attacks)

**Prompt:** `ERT_PROMPT`, a module-level constant in
`scripts/build_sealed_assets.py` (also reproduced in the paper appendix
`app:ert-prompt`).
**Runner:** `scripts/build_sealed_assets.py --stage gemini`
**Model:** `gemini-3.5-flash`, temperature 0.8, max_tokens 400
**Image:** **YES — the scene frame is attached.**
**Few-shots:** `{shots}` is filled with the first four `nominal -> ERT` pairs
from `results/phrase_artifacts/cover_release_ert_rephrases.json` (the INT-ACT /
CoVer release examples).
**Output handling:** first line of the response, quotes stripped.

**Held-out discipline:** this generator must stay distinct from whatever
generator writes the *training* attacks, or the evaluation attack distribution
leaks into training. In bridge, training attacks came from a local Qwen
(`scripts/pod4_gen_variants.py`) while evaluation attacks came from this
Gemini prompt.

**To mint more attacks per task** (we later went from 1 to 6 per task): import
`ERT_PROMPT` and reuse the same call shape — precedent in
`scripts/val8_ert_gen.py` and `scripts/gen_a29_assets.py`. Keep temperature 0.8,
dedup against the originals, keep the first N clean single-line candidates.

---

## 3. Sealed EVALUATION natural set

Two generations exist; **use the second recipe**.

**v1 (superseded, PREREG A19):** `scripts/gen_rephrase_robustness.py` —
`gemini-pro-latest`, temp 1.0, 2 sampled calls of 8 deduped to K=16,
**image attached**, a bespoke prompt that asks for natural variation but names
no axes and carries no anti-repetition. Measured outcome: one narrow register
(18 distinct sentence-initial words, lengths 6–12, 5.1 sentence templates per
16, 2% politeness). Good enough to publish, but it compresses the natural
condition and that is a stimulus artifact, not a policy property.

**v2 (current, PREREG A33):** `scripts/gen_sealed_naturals_v2.py`
- **Prompt:** the `generate.md` NATURAL block above (axes + anti-repetition).
- **Image:** **YES** for the chosen variant (`NATV2_VARIANT=image`); a text-only
  variant was generated too and kept as an artifact. The image version produced
  richer object descriptions and spatial references ("the yellow-and-green
  sponge", "the carrot on the left") and the wider length range.
- **Three authors, balanced per task:** `gemini-pro-latest` (6),
  `claude-sonnet-5` (5), `Qwen3.5-9B` pod-side (5) — all temperature 1.0, with
  the anti-repetition block carried **across authors**, so each writer sees what
  the others already produced. Author is recorded per phrase, which also enables
  a generator×applier self-match analysis later.
- **Screening:** do NOT use a lexical/word-overlap gate. It rejects legitimate
  synonyms ("ramekin" → "the white dish") and it rejected the image variant
  about twice as often as the text one, biasing the very comparison the variants
  exist to settle. Use a **semantic judge** instead
  (`scripts/judge_naturals_v2.py`, and the calibrated version used for A34):
  ask whether the candidate moves the SAME object to the SAME destination,
  explicitly allowing synonyms, politeness, questions, added description, and a
  different preposition for the same goal (`on`/`onto`/`into`/`atop`); reject
  only a different object, a different destination, or an ADJACENT relation
  (`beside`, `next to`) that no longer places the object on/in the goal.
- **Observed screening cost:** 6 of 192 dropped, **all from Qwen** (10% of its
  share); gemini 0/72 and claude 0/60. Mixing in a weaker generator widens the
  register but needs this screen.

Resulting register (192 → 186 phrases): 12.6 sentence templates per 16, lengths
5–16, 15% politeness — comparable to the bank's breadth, versus A19's 5.1 / 6–12
/ 2%.

---

## 4. Scene-description traces (what the rephraser is conditioned on)

**Prompt:** `USER_TEMPLATE` in `src/phrase_rl/cover_prompt.py` — a byte-identical
port of CoVer's template; `build_user_prompt(phrase, 1)` fills it, and
`load_system_prompt()` supplies the system half. `extract_trace(response)` cuts
the reasoning part off the reworded-list marker.
**Model:** `gemini-3.5-flash`. Temperature 0.4 for the bridge sealed assets;
0.8 in the π0.5 port.
**Image:** **YES — the scene frame is attached**, together with the phrase.

**Two rules learned the hard way:**
1. **Generate the trace FROM THE BASE PHRASE, never from the canonical.** A
   trace written from the canonical instruction smuggles the answer key into the
   input the rulebook is supposed to repair. π0.5's port promotes this to a hard
   gate: `scripts/gen_pi05_traces.py` regenerates any trace containing the
   canonical string and drops it if it persists.
2. **A trace is per (task, phrase), not per task.** New phrases need new traces.
   The registry is `results/phrase_artifacts/traces_rules_v1.parquet`
   (bridge, keyed `task,phrase`) and `traces_pi05_v1.parquet` (π0.5, keyed
   `suite,task_id,phrase`).

Also strip the trailing `Original Instruction:` block that the template appends
— it restates the canonical (`_sanitize_trace` in `scripts/rules_loop_driver.py`).

**π0.5 port, already written:** `scripts/gen_pi05_traces.py --go`.

---

## Summary table

| stage | prompt | model | temp | image? |
|---|---|---|---|---|
| bank natural + adversarial | `prompts/rules_loop/generate.md` | gemini-pro-latest | 1.0 | **no** |
| sealed adversarial (ERT) | `ERT_PROMPT` in `scripts/build_sealed_assets.py` | gemini-3.5-flash | 0.8 | **yes** |
| sealed natural v1 (A19) | bespoke, in `scripts/gen_rephrase_robustness.py` | gemini-pro-latest | 1.0 | **yes** |
| sealed natural v2 (A33) | `generate.md` NATURAL block | gemini-pro / claude-sonnet-5 / Qwen3.5-9B | 1.0 | **yes** |
| traces | `USER_TEMPLATE` in `src/phrase_rl/cover_prompt.py` | gemini-3.5-flash | 0.4–0.8 | **yes** |

All sealed-set generation is gated behind `FINAL_EVAL=1` and preflight-prints
every kept line (PREREG requirement).
