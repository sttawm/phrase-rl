"""v7 agenda items 7/8/4: arm-B duplication-rate series, step-time cost structure,
and frame-count cost projection, computed from the live v6-B telemetry log.

Run: .venv/bin/python results/analysis/armB_telemetry_v7.py [path/to/tlog6_B.jsonl]
Out: results/analysis/armB_telemetry_v7.json
"""

import json
import sys

import numpy as np

TLOG = sys.argv[1] if len(sys.argv) > 1 else (
    "/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/"
    "a45839fa-48ee-4aea-bdee-72eb4fc9dccf/scratchpad/tlog6_B.jsonl")

rows = [json.loads(l) for l in open(TLOG)]
rows.sort(key=lambda r: r["step"])
# resume dedupe: keep last occurrence per step
by_step = {r["step"]: r for r in rows}
steps = sorted(by_step)
rows = [by_step[s] for s in steps]

out = {"n_steps": len(rows), "first_step": steps[0], "last_step": steps[-1]}


def band(lo, hi):
    return [r for r in rows if lo <= r["step"] <= hi]


def mean(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None]
    return round(float(np.mean(v)), 4) if v else None


# ---- (7) duplication-rate series (dup_rate = 1 - n_unique/n_parsed analog) ----
# effective GRPO group size: n=16 sampled; unique fraction = 1 - dup_rate.
bands = [(1, 25), (26, 50), (51, 75), (76, 100), (101, 125), (126, 150), (151, 175), (176, 200)]
dup_series = {}
for lo, hi in bands:
    rs = band(lo, hi)
    if not rs:
        continue
    dup = mean(rs, "dup_rate")
    dup_series[f"{lo}-{hi}"] = {
        "dup_rate": dup,
        "eff_unique_of_16": round(16 * (1 - dup), 2) if dup is not None else None,
        "parse_fail": mean(rs, "parse_fail_rate"),
        "gate_fail": mean(rs, "gate_fail_rate"),
        "kl": mean(rs, "kl"),
        "adv_max": mean(rs, "adv_max"),
        "cand_loss_mean": mean(rs, "cand_loss_mean"),
        "n": len(rs),
    }
out["dup_bands"] = dup_series
d = [r.get("dup_rate") for r in rows if r.get("dup_rate") is not None]
s = [r["step"] for r in rows if r.get("dup_rate") is not None]
if len(d) > 10:
    A = np.vstack([s, np.ones(len(s))]).T
    slope, icpt = np.linalg.lstsq(A, d, rcond=None)[0]
    out["dup_trend"] = {"slope_per_100_steps": round(float(slope * 100), 4),
                       "intercept": round(float(icpt), 4),
                       "last25_mean": round(float(np.mean(d[-25:])), 4),
                       "first25_mean": round(float(np.mean(d[:25])), 4),
                       "max": round(float(np.max(d)), 4)}

# per-tier candidate loss trend (is ERT tier the one moving?)
tiers = {}
for tier in ("nominal", "benign", "ert"):
    tv = [(r["step"], r["cand_loss_by_tier"][tier]) for r in rows
          if r.get("cand_loss_by_tier") and tier in r["cand_loss_by_tier"]]
    if tv:
        tiers[tier] = {"first25": round(float(np.mean([v for _, v in tv[:25]])), 3),
                       "last25": round(float(np.mean([v for _, v in tv[-25:]])), 3)}
out["cand_loss_by_tier"] = tiers

# ---- (8) step-time cost structure ----
recent = rows[-30:]
tot = mean(recent, "sec")
timing = {k: mean(recent, k) for k in
          ("sec", "gen_sec", "judge_sec", "score_sec", "update_sec", "fwd_sec", "bwd_sec")}
timing["share"] = {k: round(timing[k] / tot, 3) for k in
                   ("gen_sec", "judge_sec", "score_sec", "update_sec") if timing.get(k)}
out["timing_last30"] = timing

# ---- (4) frame-count cost projection at training time ----
# score_sec covers 4-frame verifier scoring (K flow forwards + decodes per frame).
sc = timing["score_sec"]
for nf, name in [(8, "8f"), (16, "16f")]:
    extra = sc * (nf / 4 - 1)
    out[f"step_cost_{name}"] = {
        "extra_sec": round(extra, 1),
        "step_sec": round(tot + extra, 1),
        "multiplier": round((tot + extra) / tot, 3),
    }

# KL trajectory
klr = [(r["step"], r["kl"]) for r in rows if r.get("kl") is not None]
out["kl"] = {"last10_mean": round(float(np.mean([v for _, v in klr[-10:]])), 4),
             "max": round(float(np.max([v for _, v in klr])), 4)}

json.dump(out, open("results/analysis/armB_telemetry_v7.json", "w"), indent=1)
print(json.dumps(out, indent=1))
