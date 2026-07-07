"""Judge semantic faithfulness of each 0c phrase to its task's original instruction.

One Gemini call per task: original + numbered phrase list -> per-phrase verdict
(faithful = same object(s), same goal location/relation, same action). Used to
split the reward↔success correlation into faithful vs drifted subsets — the
drifted phrases are exactly where teacher-forced flow loss misranks.

  python -m phrase_rl.phase0c_faithfulness --phrases data/phrases_0c.parquet \
    --out results/phase0c/faithfulness.parquet
"""

import argparse
import json
import os

import pandas as pd
from google import genai
from google.genai import types

PROMPT = """\
Original robot task instruction: "{original}"

Below is a numbered list of candidate rephrasings. For each, judge whether it is
FAITHFUL to the original: same object(s) being manipulated, same goal location /
spatial relation, same action. A rephrase that changes the goal (e.g. "on the
plate" -> "beside the plate"), the object, or the action is NOT faithful.
Style, verb choice, and specificity changes are fine.

{numbered}

Return JSON: {{"verdicts": [{{"i": 1, "faithful": true, "reason": "..."}}, ...]}} \
with exactly {n} verdicts, reasons under 8 words."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-3.5-flash")
    args = ap.parse_args()

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    df = pd.read_parquet(args.phrases)
    rows = []
    for task, g in df.groupby("task"):
        original = g[g.arm == "original"]["phrase"].iloc[0]
        cands = g[g.arm != "original"]["phrase"].tolist()
        numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(cands))
        verdicts = None
        for temp in (0.0, 0.4, 0.7):  # malformed JSON at temp 0 is deterministic; nudge
            try:
                resp = client.models.generate_content(
                    model=args.model,
                    contents=PROMPT.format(original=original, numbered=numbered, n=len(cands)),
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=temp),
                )
                v = json.loads(resp.text)["verdicts"]
                assert len(v) == len(cands)
                verdicts = v
                break
            except (json.JSONDecodeError, AssertionError, KeyError) as e:
                print(f"{task}: parse failed at temp={temp} ({e}); retrying")
        if verdicts is None:
            raise RuntimeError(f"{task}: could not get valid verdicts")
        rows.append({"task": task, "phrase": original, "faithful": True, "reason": "original"})
        for v, p in zip(verdicts, cands):
            rows.append({"task": task, "phrase": p, "faithful": bool(v["faithful"]), "reason": v.get("reason", "")})
        n_bad = sum(not v["faithful"] for v in verdicts)
        print(f"{task}: {n_bad}/{len(cands)} judged unfaithful")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
