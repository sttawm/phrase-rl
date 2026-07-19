"""SELECTION EXAM — v7 launch gate (pre-registered BEFORE results computed).

Question: does the exam-chosen reward (C4b: 0.25*rank(z) + 0.75*rank(-grip))
work as a best-of-8 PICKER on real sampled phrase pools, against measured
rollout success? Native 4 tasks only (OOD tasks have no scorable contexts).

PRE-REGISTERED PASS RULE (declared before any number below existed):
  pooled over all screens x native tasks, C4b-picked success must
  (a) exceed the random-pick baseline (pool mean), AND
  (b) capture >= 1/3 of the (oracle - mean) selection gap.
FAIL -> v7 launch pauses for a human decision; PASS -> smoke + launch.

Pickers compared: C1 (ensemble z), grip-only, C4a (50/50), C4b (25/75).
Success labels: each sam8x1_* parquet's per-(task, phrase) success over its
24 CRN layouts. Scores: fresh same-GPU features (sel_* tables), z from the
production 4f ensemble, grip = mean grip_err; per-phrase = mean over contexts.
"""
import sys
import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from phrase_rl.verifier_reward import VerifierEnsemble  # noqa: E402

RAW = "results/overnight/raw"
K_FLOW, K_DEC = 8, 4
NATIVE = ["widowx_carrot_on_plate", "widowx_spoon_on_towel",
          "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
SCREENS = ["sam8x1_B_v6step_0120", "sam8x1_B_v6step_0160",
           "sam8x1_B_v6step_0180", "sam8x1_frozenERT"]


def norm_key(s):
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    return " ".join(s.split())


def build_rows(verbose_pq, l2_pq, amap):
    vb = pd.read_parquet(verbose_pq)
    l2 = pd.read_parquet(l2_pq)
    l2g = {k: g for k, g in l2.groupby(["task", "phrase", "episode_index", "t"])}
    rows = []
    for key, g in vb.groupby(["task", "phrase", "episode_index", "t"]):
        t, p, e, ts = key
        g = g[g.k < K_FLOW].sort_values("k")
        h = l2g.get(key)
        if h is None or len(g) < K_FLOW:
            continue
        h = h[h.k < K_DEC].sort_values("k")
        a = amap.get((t, int(e), int(ts)))
        if len(h) < K_DEC or a is None:
            continue
        rows.append({
            "task": t, "phrase": p,
            "flow_loss": g.loss.values.astype(np.float32),
            "flow_v": np.concatenate([np.asarray(x, np.float32) for x in g.v]),
            "flow_u": np.concatenate([np.asarray(x, np.float32) for x in g.u]),
            "decoded": np.concatenate([np.asarray(x, np.float32) for x in h.decoded]),
            "norm_l2": h.norm_l2.values.astype(np.float32),
            "grip_err": h.grip_err.values.astype(np.float32),
            "a_star": np.asarray(a, np.float32),
        })
    return pd.DataFrame(rows)


ast = pd.read_parquet(f"{RAW}/multit_astar.parquet")
amap = {(r.task, int(r.episode_index), int(r.t)): np.asarray(r.action_chunk, np.float32)
        for r in ast.itertuples()}
sc = pd.read_parquet(f"{RAW}/sim_contexts_eggplant.parquet")
amap.update({(r.task, int(r.episode_index), int(r.t)): np.asarray(r.action_chunk, np.float32)
             for r in sc.itertuples()})
feats = pd.concat([
    build_rows(f"{RAW}/sel_verbose_bridge.parquet", f"{RAW}/sel_l2_bridge.parquet", amap),
    build_rows(f"{RAW}/sel_verbose_egg.parquet", f"{RAW}/sel_l2_egg.parquet", amap),
], ignore_index=True)
ens = VerifierEnsemble("results/checkpoints/verifier_reward_ensemble_4f.json")
feats["z_row"] = ens.member_logits(feats).mean(axis=1)
feats["grip_row"] = [np.mean(g) for g in feats.grip_err]
feats["key"] = [norm_key(p) for p in feats.phrase]
per_phrase = feats.groupby(["task", "key"]).agg(z=("z_row", "mean"),
                                                grip=("grip_row", "mean")).reset_index()
score_map = {(r.task, r.key): (r.z, r.grip) for r in per_phrase.itertuples()}
print(f"scored phrases: {len(per_phrase)} across {per_phrase.task.nunique()} tasks")


def rank01(v):
    r = pd.Series(v).rank(method="average").values
    return (r - 1.0) / (len(v) - 1.0) if len(v) > 1 else np.array([0.5])


results = []
missing = 0
for screen in SCREENS:
    d = pd.read_parquet(f"results/val_screens/{screen}.parquet")
    for task in NATIVE:
        cells = d[d.task == task].groupby("phrase").success.mean()
        keys = [norm_key(p) for p in cells.index]
        sc_pairs = [score_map.get((task, k)) for k in keys]
        if any(p is None for p in sc_pairs):
            missing += sum(p is None for p in sc_pairs)
            keep = [i for i, p in enumerate(sc_pairs) if p is not None]
            cells = cells.iloc[keep]
            sc_pairs = [sc_pairs[i] for i in keep]
        if len(cells) < 3:
            continue
        succ = cells.values
        z = np.array([p[0] for p in sc_pairs])
        g = np.array([p[1] for p in sc_pairs])
        z01, g01 = rank01(z), rank01(-g)
        pickers = {"C1_z": z, "grip_only": -g,
                   "C4a_50_50": 0.5 * z01 + 0.5 * g01,
                   "C4b_25_75": 0.25 * z01 + 0.75 * g01}
        row = {"screen": screen, "task": task, "n": len(succ),
               "oracle": succ.max(), "mean": succ.mean()}
        for name, s in pickers.items():
            row[name] = succ[int(np.argmax(s))]
        results.append(row)

res = pd.DataFrame(results)
print(f"(phrases missing scores: {missing})")
agg = res.groupby("screen")[["oracle", "mean", "C1_z", "grip_only", "C4a_50_50", "C4b_25_75"]].mean()
print("\nper-screen (mean over native tasks), success rates:")
print((agg * 100).round(1).to_string())
pooled = res[["oracle", "mean", "C1_z", "grip_only", "C4a_50_50", "C4b_25_75"]].mean()
gap = pooled["oracle"] - pooled["mean"]
print("\nPOOLED:", (pooled * 100).round(1).to_dict())
for name in ["C1_z", "grip_only", "C4a_50_50", "C4b_25_75"]:
    cap = (pooled[name] - pooled["mean"]) / gap if gap > 0 else float("nan")
    print(f"  {name}: capture = {cap:.2f} of the selection gap")
c4b_cap = (pooled["C4b_25_75"] - pooled["mean"]) / gap
verdict = "PASS" if (pooled["C4b_25_75"] > pooled["mean"] and c4b_cap >= 1 / 3) else "FAIL"
print(f"\nPRE-REGISTERED GATE ({verdict}): C4b capture {c4b_cap:.2f} (need >=0.33 and > mean)")
res.to_json("results/analysis/selection_exam.json", orient="records", indent=1)
