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
    "originals": "originals ⇒ π0",
    "passthrough": "ERT ⇒ π0",
    "frozen_gemini_trace": "ERT + gem-trace ⇒ frozen Qwen ⇒ π0",
    "frozen_selftrace": "ERT + qwen-trace ⇒ frozen Qwen ⇒ π0",
    "sft_v2": "ERT + qwen-trace ⇒ bridge-phrase SFT'd Qwen ⇒ π0",
    "v6_rl": "ERT + gem-trace ⇒ v6-RL Qwen ⇒ π0",
    "rules_v3_gemini_trace": "ERT + gem-trace ⇒ frozen Qwen + RULES ⇒ π0",
    "rules_gemini": "ERT + gem-trace ⇒ Gemini-flash + RULES ⇒ π0",
    "gemini_bare": "ERT + gem-trace ⇒ Gemini-flash bare ⇒ π0",
    "rules_v3_selftrace": "ERT + qwen-trace ⇒ frozen Qwen + RULES ⇒ π0",
    "rules_v3_gemini_pro": "ERT + gem-trace ⇒ Gemini-pro + RULES (with reasoning) ⇒ π0",
    "gemini_pro_bare": "ERT + gem-trace ⇒ Gemini-pro bare (with reasoning) ⇒ π0",
    "claude_agent": "ERT + gem-trace ⇒ Claude agent + RULES + corpus ⇒ π0",
    "executor_pair": "ERT + gem-trace ⇒ Claude agent + RULES + corpus ⇒ π0",
    "rules_v3_claude_agent": "ERT + gem-trace ⇒ Claude agent + RULES + corpus ⇒ π0",
    "rules_pro_nominal": "nominal + gem-trace ⇒ Gemini-pro + RULES (with reasoning) ⇒ π0",
}

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

    frames = [pd.read_parquet(f) for f in sorted(glob.glob("results/sealed/*_x12.parquet"))]
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
    sb = pd.DataFrame(rows).sort_values("pooled").reset_index(drop=True)
    for _, r in sb.iterrows():
        print(f"{r.arm:26s} pooled {r.pooled:4.1f}  in-vocab {r.in_vocab:4.1f}  OOV {r.oov:4.1f}")

    ceiling = None
    conf = sorted(glob.glob("results/search/sealedsearch_*_confirm_results.json"))
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
                f"confirmed oracle ceiling {ceiling:.1f}\n(held-out, 10-task mean)",
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
