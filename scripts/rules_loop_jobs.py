#!/usr/bin/env python3
"""Pod-side executor for rules-loop jobs (run by rules_loop_worker.sh).

  score  : proxy-score (task, phrase) rows. Extracts the SAME two channels the
           proxy was calibrated on (scripts/fit_simple_success_reward.py):
             z    = -mean over CRN draws of the flow loss   (higher = better)
             grip = the server's gripper-error sidecar      (lower  = better)
           Both are consumed exactly as scripts/fine_exam_score.py does: the
           (P, K) array score_phrases returns is K LOSS DRAWS per phrase -- it
           contains no verifier column and no grip column -- and grip arrives
           out-of-band on score_phrases.last_grips.
  apply  : Qwen3.5-9B applies a rules file to each incoming phrase.

Usage (from /workspace/phrase-rl on a pod):
  .venv-gen/bin/python scripts/rules_loop_jobs.py <spec.json>

The score kind requires a running score server (the worker boots one). Context
banks: contexts_train_multit16 + contexts_val_multit16 + contexts_club (the
search's 213-instruction table). CRN: contexts per task are a seeded draw, so
every phrase in a run faces identical contexts.
"""
import json
import sys
import types
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/workspace/phrase-rl")
sys.path.insert(0, str(REPO / "src"))

spec = json.loads(Path(sys.argv[1]).read_text())
jid = spec["job_id"]
jdir = Path(sys.argv[1]).parent
payload = pd.read_parquet(jdir / f"{jid}.payload.parquet")
result_path = jdir / f"{jid}.result.parquet"

# val8 tasks are keyed by simulator task name; the context banks key training
# rows by instruction. Pool a val8 task's episodes across bank rows whose key
# contains the task stem, or it resolves to nothing and every row comes back NaN.
VAL8_STEMS = ["spoon_on_towel", "carrot_on_plate", "stack_cube", "eggplant_in_basket",
              "carrot_on_keyboard", "carrot_on_wheel", "coke_can_on_ramekin",
              "coke_can_on_plate"]


def proxy_success(z, grip, P):
    x = P["C"] + P["bz"] * np.asarray(z) + P["bg"] * (-np.asarray(grip))
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


if spec["kind"] == "score":
    from phrase_rl.phase2_train import score_phrases

    bank_files = [REPO / "data/contexts_train_multit16.parquet",
                  REPO / "data/contexts_val_multit16.parquet",
                  REPO / "data/contexts_club.parquet"]
    banks = pd.concat([pd.read_parquet(f) for f in bank_files if f.exists()],
                      ignore_index=True)
    bkey = "task" if "task" in banks.columns else "instruction"
    banks["_key"] = banks[bkey].astype(str)

    rng = np.random.default_rng(spec["seed"])
    F0 = int(spec.get("frames_per_episode", 4))
    budget = int(spec.get("score_budget", 64))
    ipc = Path("/workspace/ipc")
    # reward_mode="flow": the (P, K) return is K CRN loss draws per phrase, which
    # is what the calibration averaged. k/k_l2 match fine_exam_score's config.
    args = types.SimpleNamespace(k=8, score_seed=int(spec["seed"]), tau_min=0.0,
                                 reward_mode="flow", k_l2=0.5, score_timeout=1800)

    out = []
    for task, grp in payload.groupby("task"):
        key = str(task)
        sub = banks[banks._key == key]
        if len(sub) == 0:
            stem = next((s for s in VAL8_STEMS if s in key), None)
            if stem is not None:
                sub = banks[banks._key.str.contains(stem, regex=False)]
        phrases = list(dict.fromkeys(grp.phrase.astype(str)))
        if len(sub) == 0:
            for p in phrases:
                out.append({"task": task, "phrase": p, "z": np.nan, "grip": np.nan,
                            "n_ctx": 0, "F": 0, "C": 0})
            print(f"[warn] no contexts for {key!r} -- {len(phrases)} phrases NaN", flush=True)
            continue

        eps = sorted(sub.episode_index.unique())
        C = min(int(spec.get("contexts_per_task", 16)), len(eps))
        # C-limited tasks spend the budget on frames instead
        F = int(np.clip(round(budget / max(C, 1)), F0, 16))
        pick = rng.choice(eps, size=C, replace=False)
        ctx_rows = []
        for e in sorted(pick):
            ep = sub[sub.episode_index == e]
            ep = ep.sort_values("t") if "t" in ep.columns else ep
            ctx_rows.extend(r for _, r in ep.head(F).iterrows())

        zs, gs = [], []
        CHUNK = 8  # frame-contexts per scoring job, matching fine_exam_score
        for i in range(0, len(ctx_rows), CHUNK):
            batch = ctx_rows[i:i + CHUNK]
            contexts = [(fr, phrases) for fr in batch]
            losses = score_phrases(ipc, f"rl_{jid}_{uuid.uuid4().hex[:8]}", contexts, args)
            grips = getattr(score_phrases, "last_grips", [None] * len(losses))
            for L, G in zip(losses, grips):
                zs.append(-np.asarray(L, dtype=np.float64).mean(axis=1))
                gs.append(np.asarray(G, dtype=np.float64) if G is not None
                          else np.full(len(phrases), np.nan))
        z_mean = np.nanmean(np.stack(zs), axis=0)
        g_mean = np.nanmean(np.stack(gs), axis=0)
        for p, z, g in zip(phrases, z_mean, g_mean):
            out.append({"task": task, "phrase": p, "z": float(z), "grip": float(g),
                        "n_ctx": len(ctx_rows), "F": F, "C": C})
        print(f"{key}: {len(phrases)} phrases x F={F} C={C} "
              f"({len(ctx_rows)} frame-contexts)", flush=True)

    res = pd.DataFrame(out)
    res["proxy"] = proxy_success(res.z, res.grip, spec["proxy"])
    res.to_parquet(result_path, index=False)
    print(f"scored {len(res)} rows ({int(res.z.isna().sum())} NaN)")

elif spec["kind"] == "apply":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rules = (REPO / spec["rules_file"]).read_text()
    only = spec.get("only_rule")
    tmpl_name = "apply_single.md" if only else "apply.md"
    tmpl = (REPO / "prompts/rules_loop" / tmpl_name).read_text().replace("{{rules}}", rules)
    if only:
        tmpl = tmpl.replace("{{only_rule}}", only)

    # scene descriptions are expensive to generate: load once, reuse per phrase
    traces = {}
    tf = REPO / "results/phrase_artifacts/cover35_teacher_train.parquet"
    if tf.exists():
        t = pd.read_parquet(tf, columns=["instruction", "trace"]).drop_duplicates("instruction")
        traces = dict(zip(t.instruction.astype(str), t.trace.astype(str)))

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-9B", torch_dtype=torch.bfloat16, device_map="cuda")
    rows = []
    for r in payload.itertuples():
        trace = traces.get(str(r.task), traces.get(str(r.phrase),
                                                   "(no scene description available)"))
        prompt = (tmpl.replace("{{trace}}", trace)
                      .replace("{{phrase}}", str(r.phrase))
                      .replace("{{task}}", str(r.task)))
        inp = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(inp, do_sample=False, max_new_tokens=48)
        text = tok.decode(g[0][inp.shape[1]:], skip_special_tokens=True)
        rows.append({"task": r.task, "phrase": r.phrase,
                     "rewrite": text.strip().split("\n")[0].strip()})
    pd.DataFrame(rows).to_parquet(result_path, index=False)
    print(f"applied to {len(rows)} phrases")

else:
    raise SystemExit(f"unknown job kind {spec['kind']}")

print(f"JOB-DONE {jid} -> {result_path.name}")
