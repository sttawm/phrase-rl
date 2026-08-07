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

    banks = pd.concat([pd.read_parquet(REPO / "data/contexts_train_multit16.parquet"),
                       pd.read_parquet(REPO / "data/contexts_val_multit16.parquet")],
                      ignore_index=True)
    rng = np.random.default_rng(spec["seed"])
    ipc = Path("/workspace/ipc")
    args = types.SimpleNamespace(k=8, score_seed=spec["seed"], tau_min=0.0,
                                 reward_mode="flow", k_l2=0.5, score_timeout=1800)
    out = []
    for task, grp in payload.groupby("task"):
        sub = banks[banks.task == task] if "task" in banks.columns else \
            banks[banks.instruction == task]
        if len(sub) == 0:
            for r in grp.itertuples():
                out.append({"task": task, "phrase": r.phrase,
                            "z": np.nan, "grip": np.nan})
            continue
        eps = sorted(sub.episode_index.unique())
        pick = rng.choice(eps, size=min(spec["contexts_per_task"], len(eps)), replace=False)
        ctx_rows = [sub[sub.episode_index == e].iloc[0] for e in sorted(pick)]
        phrases = list(grp.phrase)
        contexts = [(row, phrases) for row in ctx_rows]
        losses = score_phrases(ipc, f"rl_{jid}_{abs(hash(task)) % 99999}", contexts, args)
        # losses: one (P, K) array per context; K columns = [logit..., grip...]-style
        # server output. Mean over contexts -> per-phrase channels.
        zs = np.nanmean([l[:, 0] for l in losses], axis=0)
        gs = np.nanmean([l[:, -1] for l in losses], axis=0)
        for p, z, g in zip(phrases, zs, gs):
            out.append({"task": task, "phrase": p, "z": float(z), "grip": float(g)})
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
