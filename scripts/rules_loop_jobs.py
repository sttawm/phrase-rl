#!/usr/bin/env python3
"""Pod-side executor for rules-loop jobs (run by rules_loop_worker.sh).

  score  : proxy-score (task, phrase) rows — pi0 features at C CRN contexts per
           task via the phase2 score-server IPC protocol. Emits z, grip, proxy.
  apply  : Qwen3.5-9B applies a rules file to each incoming phrase.

Usage (from /workspace/phrase-rl on a pod):
  .venv-gen/bin/python scripts/rules_loop_jobs.py <spec.json>

The score kind requires a running score server (the worker boots one). Context
banks: data/contexts_train_multit16.parquet + data/contexts_val_multit16.parquet.
CRN: contexts per task are chosen by seeded draw from the spec's seed, so every
phrase in the run is scored against the same contexts.
"""
import json
import sys
import types
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


def proxy_success(z, grip, P):
    x = P["C"] + P["bz"] * np.asarray(z) + P["bg"] * (-np.asarray(grip))
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


if spec["kind"] == "score":
    from phrase_rl.phase2_train import score_phrases

    bank_files = [REPO / "data/contexts_train_multit16.parquet",
                  REPO / "data/contexts_val_multit16.parquet",
                  REPO / "data/contexts_club.parquet"]  # the search's 213-instruction table
    banks = pd.concat([pd.read_parquet(f) for f in bank_files if f.exists()],
                      ignore_index=True)
    rng = np.random.default_rng(spec["seed"])
    F = int(spec.get("frames_per_episode", 4))
    VAL8_STEMS = ["spoon_on_towel", "carrot_on_plate", "stack_cube", "eggplant_in_basket",
                  "carrot_on_keyboard", "carrot_on_wheel", "coke_can_on_ramekin", "coke_can_on_plate"]

    ipc = Path("/workspace/ipc")
    args = types.SimpleNamespace(k=8, score_seed=spec["seed"], tau_min=0.0,
                                 reward_mode="flow", k_l2=0.5, score_timeout=1800)
    out = []
    bkey = "task" if "task" in banks.columns else "instruction"
    for task, grp in payload.groupby("task"):
        sub = banks[banks[bkey] == task]
        # val8 tasks: pool episodes across all instructions of the same stem
        if len(sub) == 0 or (spec.get("pool_val8_stems") and any(s in str(task) for s in VAL8_STEMS)):
            stem = next((s for s in VAL8_STEMS if s in str(task)), None)
            if stem is not None:
                sub = banks[banks[bkey].astype(str).str.contains(stem, regex=False)]
        if len(sub) == 0:
            for r in grp.itertuples():
                out.append({"task": task, "phrase": r.phrase,
                            "z": np.nan, "grip": np.nan})
            continue
        eps = sorted(sub.episode_index.unique())
        C = min(int(spec.get("contexts_per_task", 16)), len(eps))
        # C-limited tasks spend the budget on frames instead: F = budget/C,
        # floored at the calibration F, capped by the 16 frames the bank holds
        budget = int(spec.get("score_budget", 64))
        F = int(np.clip(round(budget / max(C, 1)), F, 16))
        pick = rng.choice(eps, size=C, replace=False)
        # F frames per picked episode, deterministic (first F by t) -- each
        # (episode, frame) row is its own context; the mean over all rows below
        # therefore averages F x C forward passes, matching the exam estimator
        ctx_rows = []
        for e in sorted(pick):
            ep_rows = sub[sub.episode_index == e].sort_values("t") if "t" in sub.columns \
                else sub[sub.episode_index == e]
            ctx_rows.extend([r for _, r in ep_rows.head(F).iterrows()])
        phrases = list(grp.phrase)
        contexts = [(row, phrases) for row in ctx_rows]
        losses = score_phrases(ipc, f"rl_{jid}_{abs(hash(task)) % 99999}", contexts, args)
        # losses: one (P, K) array per context; K columns = [logit..., grip...]-style
        # server output. Mean over contexts -> per-phrase channels.
        zs = np.nanmean([l[:, 0] for l in losses], axis=0)
        gs = np.nanmean([l[:, -1] for l in losses], axis=0)
        for p, z, g in zip(phrases, zs, gs):
            out.append({"task": task, "phrase": p, "z": float(z), "grip": float(g),
                        "n_ctx": len(ctx_rows)})
    res = pd.DataFrame(out)
    res["proxy"] = proxy_success(res.z, res.grip, spec["proxy"])
    res.to_parquet(result_path, index=False)

elif spec["kind"] == "apply":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rules = (REPO / spec["rules_file"]).read_text()
    only = spec.get("only_rule")
    if only:
        tmpl = (REPO / "prompts/rules_loop/apply_single.md").read_text()
        tmpl = tmpl.replace("{{only_rule}}", only)
    else:
        tmpl = (REPO / "prompts/rules_loop/apply.md").read_text()
    tmpl = tmpl.replace("{{rules}}", rules)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-9B", torch_dtype=torch.bfloat16, device_map="cuda")
    rows = []
    for r in payload.itertuples():
        prompt = tmpl.replace("{{phrase}}", r.phrase).replace("{{task}}", str(r.task))
        msgs = [{"role": "user", "content": prompt}]
        inp = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(inp, do_sample=False, max_new_tokens=48)
        text = tok.decode(g[0][inp.shape[1]:], skip_special_tokens=True)
        rows.append({"task": r.task, "phrase": r.phrase,
                     "rewrite": text.strip().split("\n")[0].strip()})
    pd.DataFrame(rows).to_parquet(result_path, index=False)

else:
    raise SystemExit(f"unknown job kind {spec['kind']}")

print(f"JOB-DONE {jid} -> {result_path.name}")
