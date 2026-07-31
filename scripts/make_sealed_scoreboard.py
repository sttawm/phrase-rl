#!/usr/bin/env python3
"""Sealed-test scoreboard chart: stratified 3-bar (pooled / in-vocab / OOV) per arm.

Auto-discovers every results/sealed/*_x12.parquet (multi-arm parquets contribute
each arm), strata from results/analysis/sealed_vocab_audit.json (ex-ante audit).
If row-14 confirm results exist, draws the confirmed oracle ceiling as a dashed
line (held-out layouts 18-23, mean over the 10 searched tasks — selection-free
estimate, but still a ceiling: phrases were chosen adaptively per task).

  .venv/bin/python scripts/make_sealed_scoreboard.py   (or any py with pandas+mpl)
"""
import glob
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NAME = {
    "originals": "original phrasing",
    "passthrough": "adversarial phrasing",
    "frozen_gemini_trace": "adversarial phrasing + Gemini scene description ⇒ frozen Qwen",
    "frozen_selftrace": "adversarial phrasing + self-written scene description ⇒ frozen Qwen",
    "sft_v2": "adversarial phrasing + self-written scene description ⇒ bridge-phrase SFT'd Qwen",
    "v6_rl": "adversarial phrasing + Gemini scene description ⇒ v6-RL Qwen",
    "rules_v3_gemini_trace": "adversarial phrasing + Gemini scene description ⇒ frozen Qwen + RULES",
    "rules_gemini": "adversarial phrasing + Gemini scene description ⇒ Gemini-flash + RULES",
    "gemini_bare": "adversarial phrasing + Gemini scene description ⇒ Gemini-flash bare",
    "rules_v3_selftrace": "adversarial phrasing + self-written scene description ⇒ frozen Qwen + RULES",
    "rules_v3_gemini_pro": "adversarial phrasing + Gemini scene description ⇒ Gemini-pro + RULES (with reasoning)",
    "gemini_pro_bare": "adversarial phrasing + Gemini scene description ⇒ Gemini-pro bare (with reasoning)",
    "claude_agent": "adversarial phrasing + Gemini scene description ⇒ Claude agent + RULES + corpus",
    "executor_pair": "adversarial phrasing + Gemini scene description ⇒ Claude agent + RULES + corpus",
    "rules_v3_claude_agent": "adversarial phrasing + Gemini scene description ⇒ Claude agent + RULES + corpus",
    "rules_pro_nominal": "original phrasing + Gemini scene description ⇒ Gemini-pro + RULES (with reasoning)",
    "oracle_confirmed": "oracle phrase (adaptive search) ⇒ π0  *selected on layouts 0-17",
    "rules_v4_ert": "adversarial phrasing + Gemini scene description ⇒ Gemini-pro + RULES-v4 (train-mined)",
    "rules_v4_nominal": "original phrasing + Gemini scene description ⇒ Gemini-pro + RULES-v4 (train-mined)",
    "sealed_repair": "adversarial phrasing + Gemini scene description ⇒ v7a-120 RL rewriter",
    "v7a120_polish": "original phrasing + Gemini scene description ⇒ v7a-120 RL rewriter",
    "armD_seatA_polish": "original phrasing ⇒ census router (v7a-120 → rules-v3)",
    "armD_seatA_repair": "adversarial phrasing ⇒ census router (v7a-120 → rules-v3)",
}

# arms whose full x12 payload is not on disk (aggregates live in jsonl ledgers);
# per_task dicts are equal-weight per task, matching the parquet path's task-mean
JSONL_ARMS = [
    ("results/analysis/sealed_v7a120.jsonl", {"v7a120_polish"}),
    ("results/analysis/armD_composed.jsonl", {"armD_seatA_polish", "armD_seatA_repair"}),
]

ENV_OVERRIDE = {  # audit env names that don't mechanically map to task names
    "PutGreenCubeOnPlate": "widowx_cube_on_plate_clean",
    "PutPepsiCanOnPlate": "widowx_pepsi_on_plate_clean",
}


def env_to_task(env: str) -> str:
    base = re.sub(r"InScene.*$", "", env)
    for k, v in ENV_OVERRIDE.items():
        if base.startswith(k) or k.startswith(base):
            return v
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", re.sub(r"^Put", "", base)).lower()
    return f"widowx_{s}_clean"


def main() -> None:
    audit = json.load(open("results/analysis/sealed_vocab_audit.json"))
    stratum = {env_to_task(e["env"]): e["stratum"] for e in audit}

    frames = [pd.read_parquet(f) for f in sorted(glob.glob("results/sealed/*_x12.parquet"))
              if "val8_reference" not in f]  # val-8 assets live in the same dir but are not sealed tasks
    d = pd.concat(frames, ignore_index=True)
    tasks = sorted(d.task.unique())
    missing = [t for t in tasks if t not in stratum]
    if missing:
        raise SystemExit(f"audit mapping missing tasks: {missing}")
    iv = [t for t in tasks if stratum[t] != "OOV"]
    oov = [t for t in tasks if stratum[t] == "OOV"]

    rows = []
    for arm, g in d.groupby("arm"):
        per_task = g.groupby("task").success.mean() * 100
        if len(per_task) < len(tasks):
            continue  # partial leg still mid-run
        rows.append({
            "arm": arm,
            "pooled": per_task.mean(),
            "in_vocab": per_task[iv].mean(),
            "oov": per_task[oov].mean(),
        })
    seen_jsonl = set()
    for path, wanted in JSONL_ARMS:
        try:
            lines = list(open(path))
        except FileNotFoundError:
            continue
        for l in lines:
            r = json.loads(l)
            if r.get("arm") not in wanted or r["arm"] in seen_jsonl or "per_task" not in r:
                continue
            seen_jsonl.add(r["arm"])
            pt = pd.Series(r["per_task"])
            rows.append({"arm": r["arm"], "pooled": pt.mean(),
                         "in_vocab": pt[[t for t in pt.index if t in iv]].mean(),
                         "oov": pt[[t for t in pt.index if t in oov]].mean()})
    sb = pd.DataFrame(rows).sort_values("pooled").reset_index(drop=True)
    for _, r in sb.iterrows():
        print(f"{r.arm:26s} pooled {r.pooled:4.1f}  in-vocab {r.in_vocab:4.1f}  OOV {r.oov:4.1f}")

    ceiling = None
    conf = sorted(glob.glob("results/search/sealedsearch_*_confirm_results.json"))
    if (d.arm == "oracle_confirmed").any():
        conf = []  # full-grid oracle bar exists — no separate ceiling line
    if len(conf) >= 10:
        best = [max(e["success_pct"] for e in json.load(open(f))["scoreboard"]) for f in conf]
        ceiling = sum(best) / len(best)
        print(f"confirmed oracle ceiling (held-out, {len(conf)}-task mean): {ceiling:.1f}")

    fig, ax = plt.subplots(figsize=(11, 0.52 * len(sb) + 2.2))
    y = range(len(sb))
    h = 0.26
    ax.barh([i + h for i in y], sb.pooled, h, color="#2b6cb0", label="pooled (12 tasks)")
    ax.barh(y, sb.in_vocab, h, color="#63b3ed", label=f"in-vocab ({len(iv)})")
    ax.barh([i - h for i in y], sb.oov, h, color="#f6ad55", label=f"OOV ({len(oov)})")
    for i, r in sb.iterrows():
        ax.text(r.pooled + 0.4, i + h, f"{r.pooled:.1f}", va="center", fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels([NAME.get(a, a) for a in sb.arm], fontsize=9)
    if ceiling is not None:
        ax.axvline(ceiling, ls="--", color="#822727", lw=1.4)
        ax.text(ceiling + 0.3, len(sb) - 0.5,
                f"confirmed oracle ceiling {ceiling:.1f}\n(held-out, {len(conf)}-task mean)",
                color="#822727", fontsize=8, va="top")
    ax.set_xlabel("success rate % (24 layouts × 12 reps per task)")
    ax.set_title("Sealed test scoreboard — task-mean success by pipeline")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig("results/charts/sealed_scoreboard.png", dpi=140, bbox_inches="tight", pad_inches=0.25)
    print("chart -> results/charts/sealed_scoreboard.png")


if __name__ == "__main__":
    main()
