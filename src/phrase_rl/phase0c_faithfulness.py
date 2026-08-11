"""Judge each 0c phrase against its task's original instruction — 3-class, majority-of-3.

Classes (matches the Phase 2 gate semantics):
  goal_drift — goal location/spatial relation or action changed ("on" -> "beside",
               "put on" -> "move past"). These are excluded by the training gate.
  rename     — same goal/action but object(s) referred to differently ("carrot" ->
               "the orange vegetable"). Allowed (CoVer's few-shot encourages this),
               logged as a distinct class.
  clean      — same objects, goal, action; only style/verb/specificity changed.

Majority-of-3 flash calls (temp 0.4) per task list; per-phrase class = majority,
ties resolve toward the more severe class (drift > rename > clean).

  python -m phrase_rl.phase0c_faithfulness --phrases data/phrases_0c.parquet \
    --out results/phase0c/faithfulness.parquet
"""

import argparse
import json
import os
from collections import Counter

import pandas as pd
# Lazy: faithfulness_gate imports PROMPT/SEVERITY from here on score-only pods
# that have no google-genai SDK (see faithfulness_gate.py).
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = types = None

SEVERITY = {"goal_drift": 2, "rename": 1, "clean": 0}

PROMPT = """\
Original robot task instruction: "{original}"

Below is a numbered list of candidate rephrasings. Classify EACH as exactly one of:
- "goal_drift": the goal location/spatial relation or the action changed \
(e.g. "on the plate" -> "beside the plate", "put on" -> "move past/away/toward"). \
Compound or ambiguous goals also count as goal_drift.
- "rename": goal and action preserved, but object(s) referred to by different \
words (category, color, description: "carrot" -> "the orange vegetable").
- "clean": same objects (same words or trivial determiner changes), same goal, \
same action; only style/verb/word-order/specificity changed.

{numbered}

Return JSON: {{"verdicts": [{{"i": 1, "class": "clean", "reason": "..."}}, ...]}} \
with exactly {n} verdicts, reasons under 8 words."""


def judge_once(client, model, original, cands, temp):
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(cands))
    resp = client.models.generate_content(
        model=model,
        contents=PROMPT.format(original=original, numbered=numbered, n=len(cands)),
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=temp),
    )
    v = json.loads(resp.text)["verdicts"]
    assert len(v) == len(cands)
    assert all(x["class"] in SEVERITY for x in v)
    return [x["class"] for x in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--votes", type=int, default=3)
    args = ap.parse_args()

    if genai is None:
        raise RuntimeError("google-genai is not installed in this venv")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    df = pd.read_parquet(args.phrases)
    rows = []
    for task, g in df.groupby("task"):
        original = g[g.arm == "original"]["phrase"].iloc[0]
        cands = g[g.arm != "original"]["phrase"].tolist()
        ballots = []
        for v in range(args.votes):
            for temp in (0.4, 0.7, 1.0):  # escalate on parse failure
                try:
                    ballots.append(judge_once(client, args.model, original, cands, temp))
                    break
                except (json.JSONDecodeError, AssertionError, KeyError) as e:
                    print(f"{task} vote{v}: parse fail at temp={temp}: {e}")
            else:
                raise RuntimeError(f"{task}: vote {v} failed at all temps")
        rows.append({"task": task, "phrase": original, "cls": "clean", "votes": "original"})
        for i, p in enumerate(cands):
            counts = Counter(b[i] for b in ballots)
            top = counts.most_common()
            cls = top[0][0]
            if len(top) > 1 and top[0][1] == top[1][1]:  # tie -> more severe
                cls = max((c for c, _ in top[:2]), key=lambda c: SEVERITY[c])
            rows.append({"task": task, "phrase": p, "cls": cls,
                         "votes": ",".join(b[i] for b in ballots)})
        dist = Counter(r["cls"] for r in rows if r["task"] == task and r["votes"] != "original")
        print(f"{task}: {dict(dist)}")

    out = pd.DataFrame(rows)
    out["faithful"] = out["cls"] != "goal_drift"  # Phase 2 gate semantics
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
