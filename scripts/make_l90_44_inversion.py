#!/usr/bin/env python3
"""results/charts/l90_44_inversion.png -- the trained-string inversion, one task.

"turn on the stove" is a finetuned instruction (libero_goal task 7; the LIBERO
finetune paired each task with exactly this one string). Evaluated in a NOVEL
kitchen (libero_90 task 44) the trained string fails on every init state while
every rephrasing -- natural, adversarial, and board-search-minted alike --
succeeds. Left: the same string in its trained kitchen. Right: per-phrase
success in the novel kitchen, every episode counted (tiers, screens, confirms
deduped on (phrase, init)).

  .venv/bin/python scripts/make_l90_44_inversion.py
"""
import glob
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
rows = []
for f in glob.glob(str(ROOT / "results/analysis/fourtier_live/fourtier_*.jsonl")):
    for line in open(f):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
df = pd.DataFrame(rows).drop_duplicates(subset=["suite", "task_id", "phrase", "init"], keep="last")

CANON = "turn on the stove"
g44 = df[(df.suite == "libero_90") & (df.task_id == 44)]
g7 = df[(df.suite == "libero_goal") & (df.task_id == 7) & (df.phrase == CANON)]

# tier of each phrase in the novel kitchen: canonical / natural / adversarial / search
arm_by_phrase = {}
for r in g44.itertuples():
    a = arm_by_phrase.setdefault(r.phrase, set())
    a.add(r.arm)


def kind(ph):
    arms = arm_by_phrase[ph]
    if ph == CANON:
        return "canonical"
    if any(a.startswith("nat") for a in arms):
        return "natural"
    if any(a.startswith("adv") for a in arms):
        return "adversarial"
    return "search"


per = g44.groupby("phrase").success.agg(["sum", "count"]).reset_index()
per["kind"] = per.phrase.map(kind)
per["rate"] = per["sum"] / per["count"]
KIND_ORDER = {"canonical": 0, "natural": 1, "adversarial": 2, "search": 3}
per = per.sort_values(["kind", "rate", "phrase"], key=lambda s: s.map(KIND_ORDER) if s.name == "kind" else s,
                      ascending=[True, False, True]).reset_index(drop=True)

COLORS = {"canonical": "#805ad5", "natural": "#b7791f", "adversarial": "#c53030",
          "search": "#0d9488"}
LABELS = {"canonical": "the trained string", "natural": "natural rephrasings",
          "adversarial": "adversarial rephrasings", "search": "board-search phrasings"}

fig, (aL, aR) = plt.subplots(1, 2, figsize=(15.5, 8.2), gridspec_kw={"width_ratios": [1, 2.9]})

# left: the same string where it was trained
k7, n7 = int(g7.success.sum()), len(g7)
aL.bar([0], [100.0 * k7 / max(n7, 1)], 0.55, color="#805ad5")
aL.text(0, 100.0 * k7 / max(n7, 1) + 2, "%d/%d" % (k7, n7), ha="center", fontsize=12, fontweight="bold")
aL.set_xticks([0])
aL.set_xticklabels(['"turn on the stove"\nin its TRAINED kitchen\n(libero_goal task 7)'], fontsize=10)
aL.set_ylim(0, 112)
aL.set_ylabel("rollout success %", fontsize=11)
aL.set_title("Trained scene:\nthe string works", fontsize=12)
aL.grid(alpha=0.25, axis="y")

# right: every phrase in the novel kitchen
yy = range(len(per))
aR.barh(list(yy), per.rate * 100, 0.72, color=[COLORS[k] for k in per.kind])
for i, r in per.iterrows():
    aR.text(r.rate * 100 + 1.2, i, "%d/%d" % (r["sum"], r["count"]), va="center", fontsize=8.5,
            fontweight="bold" if r.kind == "canonical" else "normal")
    label = '"%s"' % (r.phrase if len(r.phrase) <= 58 else r.phrase[:55] + "...")
    aR.text(-1.5, i, label, va="center", ha="right", fontsize=8.2,
            color=COLORS[r.kind], fontweight="bold" if r.kind == "canonical" else "normal")
aR.set_yticks([])
aR.set_xlim(0, 112)
aR.set_xlabel("rollout success % (all episodes: tiers + screens + virgin-init confirms)", fontsize=10)
aR.invert_yaxis()
aR.set_title("Novel kitchen (libero_90 task 44): the trained string fails, every rephrasing succeeds",
             fontsize=12)
aR.grid(alpha=0.25, axis="x")
aR.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in LABELS],
          labels=[LABELS[k] for k in LABELS], fontsize=9, ncol=4, loc="upper right",
          bbox_to_anchor=(1.005, -0.065), frameon=False)
aR.spines["left"].set_visible(False)
fig.subplots_adjust(left=0.06, right=0.985, wspace=0.52, bottom=0.2, top=0.9)

canon_row = per[per.kind == "canonical"].iloc[0]
reph = per[per.kind != "canonical"]
DESC = (
    'pi0.5 was finetuned on LIBERO with each task paired to exactly one instruction string. "turn on the stove" is the trained string of '
    "libero_goal task 7, where it succeeds %d/%d (left). Evaluated in a DIFFERENT kitchen (libero_90 task 44, a scene never in the finetune), "
    "the very same string scores %d/%d -- while all %d distinct rephrasings of it, from fluent requests to deliberately ornate adversarial "
    "circumlocutions to board-search candidates, together score %d/%d (right; per-phrase counts pool tier trials, 5-trial screens, and 10-trial "
    "confirms on virgin init states, deduplicated per (phrase, init)). The trained wording is bound to its training scene's behavior; ANY "
    "paraphrase escapes the binding and lets the policy ground the instruction in the scene it actually sees. The effect replicates on the only "
    "other trained-string task in our libero_90 set (task 77: canonical 1/10 vs best rephrase 8/10 on virgin confirms)."
) % (k7, n7, canon_row["sum"], canon_row["count"], len(reph), reph["sum"].sum(), reph["count"].sum())
fig.text(0.015, 0.10, DESC, fontsize=8.6, va="top", ha="left", wrap=True, color="#2d3748",
         bbox=dict(boxstyle="round,pad=0.55", fc="#f7fafc", ec="#cbd5e0", lw=0.8))

out = ROOT / "results/charts/l90_44_inversion.png"
fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.3)
print("chart -> %s" % out)
print("canonical (novel kitchen): %d/%d | rephrasings: %d/%d over %d phrases | trained kitchen: %d/%d"
      % (canon_row["sum"], canon_row["count"], reph["sum"].sum(), reph["count"].sum(), len(reph), k7, n7))
