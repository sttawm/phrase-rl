"""Phrase-evolution chart from the trainers' probe logs (type=probe rows).

Per arm x task: the greedy deployment phrase over training steps, drawn as a
timeline — a dot per probe, phrase text shown where it CHANGED, plus a rename
flag (does the phrase keep the task's object noun). Metrics panel: word count
and normalized edit distance to the original instruction.

  .venv/bin/python -m phrase_rl.probe_chart \
    --logs flow=data/probe_logs/flow.jsonl l2=data/probe_logs/l2.jsonl \
    --out results/charts/probe_evolution.png
"""

import argparse
import difflib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORIGINALS = {
    "widowx_spoon_on_towel": ("put the spoon on the towel", "spoon"),
    "widowx_carrot_on_plate": ("put carrot on plate", "carrot"),
    "widowx_stack_cube": ("stack the green block on the yellow block", "block"),
    "widowx_put_eggplant_in_basket": ("put eggplant into yellow basket", "eggplant"),
}
COLORS = {"flow": "#D85A30", "l2": "#378ADD"}


def load_probes(path):
    out = []  # (step, {task: phrase})
    for line in open(path):
        if '"type": "probe"' not in line:
            continue
        r = json.loads(line)
        out.append((r["step"], {p["task"]: p["phrase"] for p in r["probes"]}))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--out", default="results/charts/probe_evolution.png")
    args = ap.parse_args()
    arms = dict(kv.split("=", 1) for kv in args.logs)
    probes = {arm: load_probes(p) for arm, p in arms.items()}

    tasks = list(ORIGINALS)
    fig, axes = plt.subplots(len(tasks), 1, figsize=(11, 2.1 * len(tasks)), sharex=True)
    for ax, task in zip(axes, tasks):
        orig, noun = ORIGINALS[task]
        for ai, (arm, seq) in enumerate(probes.items()):
            y = 0.72 - 0.42 * ai
            last = None
            for step, d in seq:
                ph = d.get(task)
                if ph is None:
                    continue
                canon = ph.lower().strip().rstrip(".")  # ignore period/case flapping
                renamed = noun not in canon
                ax.scatter(step, y, s=46, color=COLORS.get(arm, "#888"),
                           marker="s" if renamed else "o", zorder=3)
                if canon != last:
                    ax.annotate(f"{ph}", (step, y), textcoords="offset points",
                                xytext=(4, 7 if ai == 0 else -13), fontsize=7.2,
                                color=COLORS.get(arm, "#888"))
                    last = canon
            ax.text(0.005, y, arm, transform=ax.get_yaxis_transform(),
                    fontsize=8, color=COLORS.get(arm, "#888"), va="center", ha="right")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(f"{task.replace('widowx_', '')}   (original: {orig!r})", fontsize=9.5, loc="left")
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("training step")
    axes[0].scatter([], [], marker="o", color="#666", label="keeps object noun")
    axes[0].scatter([], [], marker="s", color="#666", label="renames object")
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    fig.suptitle("Deployment-phrase evolution (greedy probe every 25 steps) — flow vs L2 arm", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"chart -> {args.out}")

    for arm, seq in probes.items():
        print(f"\n== {arm}: phrase changes ==")
        prev = {}
        for step, d in seq:
            for task, ph in d.items():
                if prev.get(task) != ph:
                    print(f"  step {step:4d}  {task.replace('widowx_',''):24s} {ph!r}")
                    prev[task] = ph


if __name__ == "__main__":
    main()
