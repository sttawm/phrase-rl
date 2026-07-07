"""CoVer's verbatim rephrase prompt — shared by the trainable Qwen and the Gemini baseline.

Prompt-parity is load-bearing (EXPERIMENT.md "Models", corrected 2026-07-07): the
headline comparison (tuned Qwen vs frontier zero-shot) must isolate WEIGHTS, so both
sides use CoVer's exact scaffold — the system persona from
reference/cover_rephrase_prompt.txt plus the user-turn template from
reference/cover_rephrase_user_template.py, which mandates inline boot-time reasoning
(image description -> instruction meaning -> noun/verb/adjective substitution lines ->
"Reworded Instructions:" numbered list) in ONE call.

USER_TEMPLATE below is a byte-identical port of the f-string CoVer's
get_rephrase_batch() produces (verified against exec'ing the reference file). That
includes its quirks: the leading newline, 12-space indentation on every line,
whitespace-only "blank" lines of exactly 12 spaces, the trailing space after
"Guidelines for generation: ", the "appeneded" typo, and the format section always
showing slots "1. / 2. / ... / {batch_number}." even when batch_number == 1. Their
code also reassigns the variable `instruction` to the built prompt — a shadowing
quirk with no effect on the emitted text (the f-string is evaluated before the
reassignment), so only the template text is ported, not the bug.

DO NOT let an editor/formatter strip trailing whitespace in this file — the template
would silently stop being verbatim. _check_template_verbatim() guards the invariants.

This module is stdlib-only on purpose: it is imported from both pod venvs
(.venv-gen for generation/training, and eval scripts wherever they run); google-genai
is imported lazily inside gemini_contents() only.

Single-phrase conditioning (p_single, Training eq. in EXPERIMENT.md):
see build_single_phrase_prefix().
"""

import argparse
import re
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Verbatim output of CoVer's get_rephrase_batch(instruction, batch_number), with the
# two f-string slots left as .format() placeholders. No literal braces exist in the
# original text, so .format() is safe.
USER_TEMPLATE = """
            Given the original instruction: "{instruction}", and the appeneded image, generate {batch_number} reworded instructions that convey the same objective.

            Guidelines for rephrasing:
            1. Use simple, clear words and actions (focus on verbs and nouns)
            2. Remove adverbs whenever possible
            3. Keep descriptions concise but complete
            4. Infer and include object colors when they can be reasonably deduced (e.g., apples are typically red, strawberries are red)
            5. Use diverse vocabulary across rephrases (vary nouns, verbs, and adjectives)
            6. Ensure each rephrase maintains the same core meaning and task objective
            7. Try to generate as diverse as possible rephrases.
            8. Consider the image when generating the rephrased instructions.
            
            Examples:
            Original: "put apple on the desk"
            Reworded: "pick up the red apple and place it on the desk", "take the apple and put it on the desk", "place the red fruit on the desk"
            
            Original: "put cooking pot in the green basket"
            Reworded: "move the silver cooking pot to the green basket", "take the cooking pot and put it in the green basket", "put the utensil into the green basket"
            
            Original: "put strawberry on top of the fridge"
            Reworded: "put the red fruit on the fridge", "place the red berry on the top of the fridge", "set the red berry on the top of the refrigerator"
            
            Original: "lift the water bottle and place it on the desk"
            Reworded: "pick up the transparent bottle and place it on the wooden desk", "take the hydration bottle and put it on the desk", "place the water on the desk"
            
            
            Guidelines for generation: 
            1. You need to consider both image and instruction when generating the rephrased instructions.
            2. You need to first generate a description of the image in your own words, and then think about what does the language instruction mean in the context of the image.
            
            
            Format your response as:
            <Description of the image>
            <Meaning of the instruction in the context of the image>
            Original: <Nouns> as many as possible potential replacements: <Nouns>
            Original: <Verbs> as many as possible potential replacements: <Verbs>
            Original: <Adjectives> as many as possible potential replacements: <Adjectives>
            Original: <Adverbs>

            Original Instruction:
            {instruction}

            Reworded Instructions:
            1. <Alternative phrasing 1>
            2. <Alternative phrasing 2>
            ...
            {batch_number}. <Alternative phrasing {batch_number}>
            
            Important: Ensure all rephrased instructions avoid adverbs, use diverse vocabulary, and maintain the same objective as the original.
            """


def _check_template_verbatim():
    """Trip loudly if the significant whitespace above was ever normalized away."""
    lines = USER_TEMPLATE.splitlines()
    ok = (
        len(lines) == 51
        and sum(1 for ln in lines if ln == " " * 12) == 10  # 12-space "blank" lines
        and sum(1 for ln in lines if ln == "") == 4  # truly empty lines
        and "            Guidelines for generation: " in lines  # trailing space
        and USER_TEMPLATE.startswith("\n            Given the original instruction")
        and USER_TEMPLATE.endswith("objective as the original.\n            ")
    )
    if not ok:
        raise RuntimeError(
            "USER_TEMPLATE no longer matches CoVer's verbatim template — significant "
            "whitespace was probably stripped by an editor/formatter. Re-derive it from "
            "reference/cover_rephrase_user_template.py."
        )


_check_template_verbatim()


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    """CoVer's system persona, verbatim from reference/cover_rephrase_prompt.txt."""
    return (_REPO_ROOT / "reference" / "cover_rephrase_prompt.txt").read_text()


def build_user_prompt(instruction: str, batch_number: int) -> str:
    """CoVer's verbatim user turn asking for `batch_number` reworded instructions."""
    return USER_TEMPLATE.format(instruction=instruction, batch_number=batch_number)


_MARKER_RE = re.compile(r"reworded\s+instructions\s*:?", re.I)
_ITEM_RE = re.compile(r"^\s*\d+\s*[.)]\s+(\S.*?)\s*$")


def _clean_item(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":  # models sometimes quote items
        s = s[1:-1].strip()
    return s


def parse_reworded(text: str) -> list[str]:
    """Extract the final numbered rephrase list from a CoVer-format response.

    Port of CoVer's extract_reworded_instructions with defenses:
    - marker = LAST line containing "Reworded Instructions" (colon optional, any
      case) — models sometimes echo the scaffold's format section, which contains
      an earlier copy of the marker;
    - items may start on the marker line itself ("Reworded Instructions: 1. ...");
    - "1." and "1)" numbering both accepted; whitespace and wrapping quotes stripped;
    - blank lines between items are skipped; collection stops at the first non-empty
      non-numbered line (post-list commentary);
    - fallback when no marker (or nothing after it): ALL numbered lines after the
      LAST line ending in ":"; else [].
    """
    if "</think>" in text:  # thinking is disabled everywhere; defensive anyway
        text = text.rsplit("</think>", 1)[-1]
    lines = text.splitlines()

    marker_idx, marker_end = None, 0
    for i, ln in enumerate(lines):
        m = _MARKER_RE.search(ln)
        if m:
            marker_idx, marker_end = i, m.end()

    items = []
    if marker_idx is not None:
        remainder = lines[marker_idx][marker_end:].strip()
        m = _ITEM_RE.match(remainder)
        if m:
            items.append(_clean_item(m.group(1)))
        for ln in lines[marker_idx + 1 :]:
            if not ln.strip():
                continue
            m = _ITEM_RE.match(ln)
            if not m:
                break
            items.append(_clean_item(m.group(1)))
        if items:
            return items

    # defensive fallback: all numbered lines after the last line ending in ":"
    colon_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s and s.endswith(":"):
            colon_idx = i
    if colon_idx is None:
        return []
    for ln in lines[colon_idx + 1 :]:
        m = _ITEM_RE.match(ln)
        if m:
            items.append(_clean_item(m.group(1)))
    return items


def build_qwen_messages(image, instruction: str, batch_number: int) -> list:
    """Chat messages for Qwen3.5 under the CoVer scaffold.

    Feed to processor.apply_chat_template(..., add_generation_prompt=True,
    enable_thinking=False) — thinking MUST stay disabled (EXPERIMENT.md "Models").
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": load_system_prompt()}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_user_prompt(instruction, batch_number)},
            ],
        },
    ]


def build_single_phrase_prefix(instruction: str, image) -> list:
    """The p_single conditioning for the training objective.

    Returns the FULL chat messages list, ending in an already-begun assistant turn:
        [system(load_system_prompt()),
         user([image, build_user_prompt(instruction, 1)]),
         assistant "1. "]
    Feed to processor.apply_chat_template(..., continue_final_message=True,
    enable_thinking=False) so the "1. " assistant prefix is kept open for the
    candidate phrase to continue it.

    Definition of log pi_theta(y | c, p_single) used by the advantage-weighted loss
    (EXPERIMENT.md "Training" step 1): tokenize the chat template up to and
    including the assistant prefix "1. " (enable_thinking=False, no reasoning
    tokens), then score the candidate phrase y as the continuation, terminated by
    "\\n" (or EOS). Loss/credit applies to the tokens of y (+ terminator) ONLY —
    never to the prompt, the "1. " prefix, or any inline reasoning, which is
    deliberately absent from this conditioning: p_single treats y as the first
    list item emitted with no preceding CoT, so every candidate is scored under
    the identical, reasoning-free prefix regardless of what reasoning happened to
    precede it at sampling time.

    Note the batch_number=1 template still shows the "1. / 2. / ... / 1." format
    slots — that is CoVer's own N=1 rendering, kept verbatim.
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": load_system_prompt()}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_user_prompt(instruction, 1)},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "1. "}],
        },
    ]


def gemini_contents(image_png_bytes: bytes, instruction: str, batch_number: int) -> list:
    """contents list for google-genai generate_content (frontier eval baseline).

    Pass load_system_prompt() as GenerateContentConfig(system_instruction=...) —
    CoVer runs the persona as the system turn, not inline. CoVer used temp 0.8.
    """
    from google.genai import types  # lazy: google-genai not installed in every venv

    return [
        types.Part.from_bytes(data=image_png_bytes, mime_type="image/png"),
        build_user_prompt(instruction, batch_number),
    ]


def main():
    """Eyeball helper: print the exact prompts for an instruction."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default="put the carrot on the plate")
    ap.add_argument("--batch-number", type=int, default=16)
    args = ap.parse_args()
    print("=== system ===")
    print(load_system_prompt())
    print("=== user ===")
    print(build_user_prompt(args.instruction, args.batch_number))
    print("=== p_single === system + user(image, batch_number=1 turn) + assistant '1. ' "
          "(see build_single_phrase_prefix)")


if __name__ == "__main__":
    main()
