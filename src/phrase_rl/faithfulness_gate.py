"""In-training-loop faithfulness gate: cached, retry-hardened 3-class LLM judge.

Phase 0c confirmed the reward-hacking axis: goal_drift phrases (goal location /
spatial relation / action changed) are reward-favored AND success-poor, while
renames ("carrot" -> "the orange vegetable") are behaviorally fine (highest
success in 0c) and explicitly encouraged by CoVer's few-shot examples. The
Phase 2 trainer therefore computes advantages only among gate-passing
candidates. This module wraps the exact phase0c judge (same PROMPT, same
SEVERITY, majority-of-votes, ties resolve to the more severe class) as an
async training-loop component:

  - Disk cache keyed by sha256(original \\x00 candidate): phrase distributions
    early in RL repeat heavily, so re-judging identical phrases across steps
    and resumes is free. The key deliberately ignores judge model / vote count
    — wipe or repoint the cache if you change judges. Saved atomically
    (tmp + os.replace) after every fresh judgment; a crash never loses paid
    API results.
  - Retry/backoff + hard-stop hygiene copied from gemini_rephrase: server
    retry delays are honored, and a depleted-credit 429 raises GateUnavailable
    so the trainer can checkpoint and pause rather than train ungated.
  - Fail-closed: a candidate whose votes ALL fail (parse errors, exhausted
    retries) gets cls="judge_fail", faithful=False, and is NOT cached (it will
    be re-judged on next sight). For the three real classes
    faithful == (cls != "goal_drift"). Partial ballots (some votes failed) are
    used for this call but not cached either — only full-vote verdicts persist.

Runs in .venv-gen (google-genai + pandas; no torch). Manual smoke:
  .venv-gen/bin/python -m phrase_rl.faithfulness_gate \
    --original "put the carrot on the plate" \
    --candidates "place the carrot on the plate" "move the carrot off the table"
"""

import argparse
import asyncio
import hashlib
import json
import re
import os
from collections import Counter

from google import genai
from google.genai import types
from tqdm import tqdm

from phrase_rl.gemini_rephrase import HARD_STOP_MARKERS, server_retry_delay
from phrase_rl.phase0c_faithfulness import PROMPT, SEVERITY

TEMPS = (0.4, 0.7, 1.0)  # escalate temperature on parse failure, as in phase0c


class GateUnavailable(RuntimeError):
    """The judge cannot run (depleted quota / dead billing / preflight failure).

    The trainer must catch this, checkpoint, and pause — never train ungated.
    """


def _key(original: str, candidate: str) -> str:
    return hashlib.sha256(f"{original}\x00{candidate}".encode()).hexdigest()


def _majority(votes: list[str]) -> str:
    """Majority class; ties (incl. three-way) resolve to the most severe."""
    counts = Counter(votes)
    top = max(counts.values())
    tied = [c for c, n in counts.items() if n == top]
    return max(tied, key=lambda c: SEVERITY[c])


class FaithfulnessGate:
    """Majority-of-votes 3-class judge (clean / rename / goal_drift) with disk cache.

    Counters (for the trainer to log; also see .stats()):
      n_judged      — unique candidates judged fresh via the API
      n_cache_hits  — unique candidates answered from the disk cache
      class_counts  — histogram over every verdict returned by judge()
    """

    def __init__(self, model: str = "gemini-3.1-flash-lite", votes: int = 1,
                 api_key: str | None = None, cache_path: str = "data/gate_cache.json",
                 retries: int = 6, generate_fn=None, include_reasons: bool = True):
        """generate_fn: optional local backend — a sync callable (prompt_text) -> raw
        response text (e.g. the frozen base Qwen via adapter-disabled generate).
        When set, no Gemini client/key is needed and GateUnavailable cannot occur
        from quota. JSON-mode formatting is requested in-prompt instead."""
        self.model = model if generate_fn is None else "local"
        self.votes = votes
        self.generate_fn = generate_fn
        self.include_reasons = include_reasons
        self.api_key = None
        if generate_fn is None:
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise RuntimeError("no api_key given and GEMINI_API_KEY not set")
        self.retries = retries
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    self.cache = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"corrupt gate cache {cache_path} ({e}) — inspect/delete it "
                    "manually; refusing to silently overwrite paid judgments"
                )
        if os.path.dirname(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self.n_judged = 0
        self.n_cache_hits = 0
        self.class_counts: Counter = Counter()
        # client is (re)created per event loop: judge_many_sync uses asyncio.run,
        # and the SDK's async transport must not outlive the loop it bound to
        self._client = None
        self._client_loop = None
        self._preflighted = False
        self._hard_stop = asyncio.Event()

    # ---------------------------------------------------------------- client

    def _get_client(self):
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            self._client = genai.Client(api_key=self.api_key)
            self._client_loop = loop
        return self._client

    async def preflight(self):
        if self.generate_fn is not None:
            return  # local backend: nothing to preflight
        """One tiny call so a dead quota/billing state fails in seconds, not
        mid-training after launching a batch of doomed, retry-storming calls."""
        try:
            await self._get_client().aio.models.generate_content(model=self.model, contents="ok")
        except Exception as e:
            raise GateUnavailable(f"preflight failed for {self.model}: {str(e)[:300]}")
        self._preflighted = True

    # ---------------------------------------------------------------- voting

    async def _vote_once(self, original: str, cands: list[str], temp: float) -> list[str]:
        numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(cands))
        base_prompt = PROMPT.format(original=original, numbered=numbered, n=len(cands))
        if not self.include_reasons:  # in-loop: classes only, ~3x fewer output tokens
            base_prompt = base_prompt.replace(
                '{"verdicts": [{"i": 1, "class": "clean", "reason": "..."}, ...]}',
                '{"verdicts": [{"i": 1, "class": "clean"}, ...]}').replace(
                ", reasons under 8 words", "")
        if self.generate_fn is not None:
            prompt = base_prompt + \
                "\nRespond with ONLY the JSON object, no code fences, no commentary."
            text = await asyncio.to_thread(self.generate_fn, prompt)
            m = re.search(r"\{.*\}", text, flags=re.S)  # tolerate pre/post chatter
            if not m:
                raise ValueError(f"no JSON object in local judge output: {text[:120]!r}")
            verdicts = json.loads(m.group(0))["verdicts"]
            if len(verdicts) != len(cands):
                raise ValueError(f"{len(verdicts)} verdicts for {len(cands)} candidates")
            classes = [v["class"] for v in verdicts]
            bad = [c for c in classes if c not in SEVERITY]
            if bad:
                raise ValueError(f"unknown class(es): {bad}")
            return classes
        resp = await self._get_client().aio.models.generate_content(
            model=self.model,
            contents=base_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=temp,
                # thinking OFF: verdicts are ~30 tokens; default thinking billed 1-2k
                # output tokens per call (2026-07-10: drained the Gemini credits)
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        verdicts = json.loads(resp.text)["verdicts"]
        if len(verdicts) != len(cands):
            raise ValueError(f"{len(verdicts)} verdicts for {len(cands)} candidates")
        classes = [v["class"] for v in verdicts]
        bad = [c for c in classes if c not in SEVERITY]
        if bad:
            raise ValueError(f"unknown class(es): {bad}")
        return classes

    async def _vote(self, original: str, cands: list[str]) -> list[str] | None:
        """One ballot with parse-escalation + backoff. None = ballot failed
        (retries exhausted). Raises GateUnavailable on unretryable quota state."""
        parse_fails, last = 0, "?"
        for attempt in range(self.retries):
            if self._hard_stop.is_set():
                raise GateUnavailable("aborted: hard stop already triggered")
            try:
                temp = TEMPS[min(parse_fails, len(TEMPS) - 1)]
                return await self._vote_once(original, cands, temp)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                parse_fails += 1  # bad JSON/schema: retry hotter, no backoff
                last = f"parse: {str(e)[:150]}"
            except Exception as e:
                msg = str(e)
                if "429" in msg and any(m in msg.lower() for m in HARD_STOP_MARKERS):
                    self._hard_stop.set()
                    raise GateUnavailable(f"unretryable quota state: {msg[:200]}")
                last = msg[:200]
                delay = server_retry_delay(msg) or min(15 * 2**attempt, 90)
                try:  # sleep, but abort immediately if a sibling hard-stops
                    await asyncio.wait_for(self._hard_stop.wait(), timeout=delay + 1)
                    raise GateUnavailable("hard stop during backoff")
                except asyncio.TimeoutError:
                    pass
        print(f"gate: ballot failed after {self.retries} attempts ({last})")
        return None

    # ------------------------------------------------------------------ API

    async def judge(self, original: str, candidates: list[str]) -> list[dict]:
        """Judge candidates against the original. Returns, per candidate (in
        order): {"cls", "faithful", "votes", "cached"}. One API call per ballot
        covers all uncached candidates (numbered list, as in phase0c)."""
        if self._hard_stop.is_set():
            raise GateUnavailable("hard stop set — refusing to return unjudged results")
        verdicts: dict[str, dict] = {}
        todo = []
        for cand in candidates:
            if cand in verdicts or cand in todo:  # duplicate within this call
                continue
            hit = self.cache.get(_key(original, cand))
            if hit is not None:
                self.n_cache_hits += 1
                verdicts[cand] = {"cls": hit["cls"], "votes": hit["votes"], "cached": True}
            else:
                todo.append(cand)

        if todo:
            ballots = []
            for _ in range(self.votes):
                b = await self._vote(original, todo)
                if b is not None:
                    ballots.append(b)
            full = len(ballots) == self.votes
            for i, cand in enumerate(todo):
                self.n_judged += 1
                if ballots:
                    cls = _majority([b[i] for b in ballots])
                    votes_str = ",".join(b[i] for b in ballots)
                    verdicts[cand] = {"cls": cls, "votes": votes_str, "cached": False}
                    if full:  # only full-confidence verdicts persist
                        self.cache[_key(original, cand)] = {
                            "original": original, "candidate": cand,
                            "cls": cls, "votes": votes_str,
                        }
                else:  # zero ballots: fail closed, not cached
                    verdicts[cand] = {"cls": "judge_fail", "votes": "", "cached": False}
            if full:
                self._save_cache()

        out = []
        for cand in candidates:
            v = verdicts[cand]
            self.class_counts[v["cls"]] += 1
            out.append({"cls": v["cls"], "faithful": v["cls"] in ("clean", "rename"),
                        "votes": v["votes"], "cached": v["cached"]})
        return out

    async def judge_many(self, items: list[tuple[str, list[str]]],
                         concurrency: int = 4, progress: bool = True) -> list[list[dict]]:
        """Judge a batch of (original, candidates) with bounded concurrency.
        Results align with input order. Raises GateUnavailable if the quota
        hard-stops mid-batch (partial results are already cached to disk)."""
        if not self._preflighted:
            await self.preflight()
        self._hard_stop = asyncio.Event()  # fresh run: allow recovery after a refill
        sem = asyncio.Semaphore(concurrency)

        async def one(i, original, cands):
            async with sem:
                return i, await self.judge(original, cands)

        tasks = [asyncio.create_task(one(i, o, c)) for i, (o, c) in enumerate(items)]
        results: list = [None] * len(items)
        hard_err = None
        it = asyncio.as_completed(tasks)
        if progress:
            it = tqdm(it, total=len(tasks), desc="faithfulness gate", leave=False)
        for fut in it:  # drain everything so no in-flight task leaks
            try:
                i, res = await fut
                results[i] = res
            except GateUnavailable as e:
                hard_err = hard_err or e
        if hard_err is not None:
            raise hard_err
        return results

    def judge_many_sync(self, items: list[tuple[str, list[str]]],
                        concurrency: int = 4, progress: bool = True) -> list[list[dict]]:
        return asyncio.run(self.judge_many(items, concurrency=concurrency, progress=progress))

    # ------------------------------------------------------------ bookkeeping

    def stats(self) -> dict:
        return {"n_judged": self.n_judged, "n_cache_hits": self.n_cache_hits,
                "class_counts": dict(self.class_counts), "cache_size": len(self.cache)}

    def _save_cache(self):
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.cache, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.cache_path)


def main():
    ap = argparse.ArgumentParser(description="manual smoke test for the faithfulness gate")
    ap.add_argument("--original", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--cache", default="data/gate_cache.json")
    args = ap.parse_args()

    gate = FaithfulnessGate(model=args.model, votes=args.votes, cache_path=args.cache)
    (res,) = gate.judge_many_sync([(args.original, args.candidates)])
    for cand, v in zip(args.candidates, res):
        print(f"{v['cls']:>10}  faithful={str(v['faithful']):5}  "
              f"votes=[{v['votes']}]  cached={v['cached']}  {cand!r}")
    print(gate.stats())


if __name__ == "__main__":
    main()
