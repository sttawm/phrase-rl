"""Step-0 A/B verdict: inline vs Gemini-trace conditioning (+ CoVer-Gemini baseline).

Inputs: scores_0b_redo.parquet (CRN losses for the three CoVer-template arms) and
the faithfulness gate cache (drift classes; judged calls are cache hits by now).
Per arm: reliability at K, reward spread, drift/rename rates, oracle best-of-N
vs original — overall AND gate-pass-only (the set RL actually trains on).
Outputs: clean 4-panel chart + json + a printed recommendation.

  python -m phrase_rl.ab_verdict --scores data/scores_0b_redo.parquet
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phrase_rl.faithfulness_gate import FaithfulnessGate, GateUnavailable
from phrase_rl.phase0b_analyze import split_half_stats

ARM_LABELS = {"cover_qwen_inline": "Qwen inline\n(own reasoning)",
              "cover_qwen_trace": "Qwen + Gemini\ntrace",
              "cover_gemini": "Gemini\n(frontier baseline)"}
GATE_DEAD = False
COLORS = {"cover_qwen_inline": "#85B7EB", "cover_qwen_trace": "#7F77DD", "cover_gemini": "#F0997B"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/scores_0b_redo.parquet")
    ap.add_argument("--out-json", default="results/overnight/ab_verdict.json")
    ap.add_argument("--out-chart", default="results/charts/step0_ab_verdict.png")
    ap.add_argument("--no-judge", action="store_true", help="skip gate entirely (credits out)")
    args = ap.parse_args()

    global GATE_DEAD
    if args.no_judge:
        GATE_DEAD = True
    sc = pd.read_parquet(args.scores)
    gate = None if args.no_judge else FaithfulnessGate(votes=3, cache_path="data/gate_cache.json")
    arms = [a for a in sc["arm"].unique() if a != "original"]
    orig_loss = sc[sc.arm == "original"].groupby(["episode_index", "t"])["loss"].mean()

    res = {}
    for arm in arms:
        a = sc[sc.arm == arm]
        rows = []
        per_ctx = []
        for (ep, t), g in a.groupby(["episode_index", "t"]):
            mat = g.pivot_table(index="phrase", columns="k", values="loss")
            if len(mat) < 8:
                continue
            rho, sig, _ = split_half_stats(mat.to_numpy(), zscore=True)
            phrases = list(mat.index)
            instruction_rows = sc[(sc.episode_index == ep) & (sc.t == t) & (sc.arm == "original")]
            original = instruction_rows["phrase"].iloc[0] if len(instruction_rows) else None
            cls = {}
            if original and not GATE_DEAD:
                try:
                    (verdicts,) = gate.judge_many_sync([(original, phrases)], progress=False)
                    cls = {p: v["cls"] for p, v in zip(phrases, verdicts)}
                except GateUnavailable:
                    GATE_DEAD = True  # circuit-break: stop attempting; judge-free metrics only
            means = mat.mean(axis=1)
            pass_means = means[[p for p in phrases if cls.get(p) not in ("goal_drift", "judge_fail")]]
            o = orig_loss.get((ep, t), np.nan)
            per_ctx.append(dict(
                rho=rho, spread=float(means.std()),
                spread_pass=float(pass_means.std()) if (cls and len(pass_means) > 3) else np.nan,
                drift=np.mean([cls.get(p) == "goal_drift" for p in phrases]) if cls else np.nan,
                rename=np.mean([cls.get(p) == "rename" for p in phrases]) if cls else np.nan,
                oracle_gain=float((o - means.min()) / o) if o == o else np.nan,
                oracle_gain_pass=float((o - pass_means.min()) / o) if (cls and o == o and len(pass_means) > 3) else np.nan,
            ))
        d = pd.DataFrame(per_ctx)
        res[arm] = {k: round(float(np.nanmedian(d[k])), 4) for k in d.columns}
        res[arm]["n_contexts"] = len(d)
        res[arm]["n_judged"] = int(d["drift"].notna().sum())

    # decision heuristic: prefer the arm with lower drift and >= gate-pass oracle gain
    qi, qt = res.get("cover_qwen_inline", {}), res.get("cover_qwen_trace", {})
    if qi and qt:
        trace_wins = (qt.get("drift", 1) < qi.get("drift", 1) - 0.02 and
                      qt.get("oracle_gain_pass", 0) >= qi.get("oracle_gain_pass", 0) - 0.02)
        res["recommendation"] = ("trace" if trace_wins else "inline")
        res["recommendation_note"] = (
            "trace wins: materially lower drift without losing gate-pass headroom"
            if trace_wins else
            "inline holds: trace does not reduce drift materially and/or costs headroom; "
            "inline keeps prompt parity + no frontier call")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(res, open(args.out_json, "w"), indent=2)

    all_metrics = [("drift", "goal-drift rate\n(lower better)"), ("rho", "reliability ρ at K=16"),
                   ("oracle_gain_pass", "gate-pass oracle gain\nvs original"), ("spread_pass", "gate-pass reward spread"),
                   ("oracle_gain", "oracle gain vs original\n(all candidates)"), ("spread", "reward spread\n(all candidates)")]
    metrics = [(m, t) for m, t in all_metrics
               if any(res[a].get(m) == res[a].get(m) for a in arms)][:4]  # drop all-NaN panels
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.8 * len(metrics), 3.6))
    for ax, (m, title) in zip(axes, metrics):
        vals = [res[a].get(m, np.nan) for a in arms]
        ax.bar(range(len(arms)), vals, color=[COLORS.get(a, "#888") for a in arms])
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([ARM_LABELS.get(a, a) for a in arms], fontsize=8)
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(vals):
            if v == v: ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.suptitle("Step-0 A/B — conditioning arms on the CoVer template (120 val contexts, median per context)", y=1.04)
    fig.tight_layout()
    fig.savefig(args.out_chart, dpi=130, bbox_inches="tight")
    print(json.dumps(res, indent=2))
    print(f"\nchart -> {args.out_chart}")


if __name__ == "__main__":
    main()
