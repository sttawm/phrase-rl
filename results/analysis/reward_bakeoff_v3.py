"""v7 Step 3 FULL Gate-Zero exam: candidate bake-off on FRESH same-GPU features.

Difference from reward_bakeoff_v2.py (which ran on STORED study tables covering
only 4/68 native pairs): ALL phrase features here come from the fresh examfull
tables extracted in one GPU pass over every powered exam phrase:

  results/overnight/raw/examfull_verbose_bridge.parquet  carrot/spoon/stack,
  results/overnight/raw/examfull_l2_bridge.parquet       200 Bridge quartile
                                                         contexts (multit_astar)
  results/overnight/raw/examfull_verbose_egg.parquet     eggplant, 60 SIM
  results/overnight/raw/examfull_l2_egg.parquet          contexts (sim_contexts_
                                                         eggplant; sim caveat on
                                                         learned fine readouts)

Candidate implementations are REUSED VERBATIM from v2 (frozen before this run):
norm_cov/norm_key, the 4f ensemble z, row-mean grip, gate = z >= within-task
median AND z >= frozen passthrough FLOOR (floor derived from Bridge nominals
only, applies everywhere), winsor p5/p95, rank01 blends, echo penalty with the
frozen lexicon. ONE forced semantic change, documented: the within-task scored
group (medians / ranks / winsor bounds) is now the task's EXAM phrase set
(9/11/8/8) rather than v2's 35-phrase study group, because all features are
sourced from the examfull tables per the v3 protocol.

Exam artifacts (frozen, never edited):
  results/analysis/gate_zero_pairs.json          139 pairs (68 native, 71 OOD)
  results/analysis/phrase_success_table.parquet  success cells + clustered se

Selection rule (pre-registered, REWARD_PLAN.md Step 3): pass ALL covered native
gate-zero signs -> among passers lowest max top-1 regret, CI-cleared vs the 6pp
cap -> tie-break leave-task-out transfer of the blend weight.

Run:  .venv/bin/python results/analysis/reward_bakeoff_v3.py
Out:  results/analysis/reward_bakeoff_v3.json
"""

import itertools
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

REPO = "/Users/sttawm/dev/robotics/phrase-rl"
os.chdir(REPO)
sys.path.insert(0, f"{REPO}/src")
RAW = "results/overnight/raw"
K_FLOW, K_DEC = 8, 4
B = 2000
REGRET_CAP_PP = 6.0
RNG = np.random.default_rng(0)

from phrase_rl.verifier_reward import VerifierEnsemble  # noqa: E402

NATIVE = ["widowx_carrot_on_plate", "widowx_spoon_on_towel",
          "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
NOMINAL = {
    "widowx_carrot_on_plate": "put carrot on plate",
    "widowx_spoon_on_towel": "put the spoon on the towel",
    "widowx_stack_cube": "stack the green block on the yellow block",
    "widowx_put_eggplant_in_basket": "put eggplant into yellow basket",
    "widowx_coke_can_on_plate_clean": "put coke can on plate",
    "widowx_coke_can_on_ramekin_clean": "put coke can on ramekin",
    "widowx_carrot_on_wheel_clean": "put carrot on wheel",
    "widowx_carrot_on_keyboard_clean": "put carrot on keyboard",
}
LEXICON = {"center", "middle", "exactly", "precisely", "vertically", "upright",
           "gently", "directly", "carefully", "neatly", "firmly", "slowly"}
LAMBDAS = (0.5, 1.0)
W_GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]  # LTO diagnostic
W_PREREG = [0.25, 0.5]                                            # the C4 family


def norm_cov(s):
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    return re.sub(r"\s+", " ", s).strip()


def norm_key(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", norm_cov(s))).strip()


def toks(s):
    return set(re.findall(r"[a-z]+", norm_cov(s)))


def echo_count(text, task):
    return len((toks(text) & LEXICON) - toks(NOMINAL[task]))


# ---------------- feature rows (identical to v2) ----------------

def build_rows(verbose_pq, l2_pq, amap, source):
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
            "task": t, "phrase": p, "episode_index": int(e), "t": int(ts), "source": source,
            "flow_loss": g.loss.values.astype(np.float32),
            "flow_v": np.concatenate([np.asarray(x, np.float32) for x in g.v]),
            "flow_u": np.concatenate([np.asarray(x, np.float32) for x in g.u]),
            "decoded": np.concatenate([np.asarray(x, np.float32) for x in h.decoded]),
            "norm_l2": h.norm_l2.values.astype(np.float32),
            "grip_err": h.grip_err.values.astype(np.float32),
            "a_star": np.asarray(a, np.float32),
        })
    return pd.DataFrame(rows)


def load_features():
    ast = pd.read_parquet(f"{RAW}/multit_astar.parquet")
    amap = {(r.task, int(r.episode_index), int(r.t)): np.asarray(r.action_chunk, np.float32)
            for r in ast.itertuples()}
    bridge = build_rows(f"{RAW}/examfull_verbose_bridge.parquet",
                        f"{RAW}/examfull_l2_bridge.parquet", amap, "examfull_bridge")
    sc = pd.read_parquet(f"{RAW}/sim_contexts_eggplant.parquet")
    amap_s = {(r.task, int(r.episode_index), int(r.t)): np.asarray(r.action_chunk, np.float32)
              for r in sc.itertuples()}
    egg = build_rows(f"{RAW}/examfull_verbose_egg.parquet",
                     f"{RAW}/examfull_l2_egg.parquet", amap_s, "examfull_egg_sim")
    return pd.concat([bridge, egg], ignore_index=True)


def load_stored_features():
    """v2's stored-table loader -- DIAGNOSTIC ONLY (fresh-vs-stored agreement)."""
    ast = pd.read_parquet(f"{RAW}/multit_astar.parquet")
    amap = {(r.task, int(r.episode_index), int(r.t)): np.asarray(r.action_chunk, np.float32)
            for r in ast.itertuples()}
    multit = build_rows(f"{RAW}/study_multit_verbose.parquet",
                        f"{RAW}/study_multit_l2.parquet", amap, "study_multit")
    sc = pd.read_parquet(f"{RAW}/sim_contexts_eggplant.parquet")
    amap_s = {(r.task, int(r.episode_index), int(r.t)): np.asarray(r.action_chunk, np.float32)
              for r in sc.itertuples()}
    sim = build_rows(f"{RAW}/eggplant_sim_verbose.parquet",
                     f"{RAW}/eggplant_sim_l2.parquet", amap_s, "eggplant_sim")
    return pd.concat([multit, sim], ignore_index=True)


# ---------------- per-task scored state (identical to v2) ----------------

class TaskState:
    """Per-task pivot matrices over (key, episode) + frozen per-key metadata."""

    def __init__(self, task, sub):
        self.task = task
        self.source = sub.source.iloc[0]
        sub = sub.assign(key=[norm_key(p) for p in sub.phrase])
        self.keys = sorted(sub.key.unique())
        self.raw_text = {k: sorted(sub[sub.key == k].phrase.unique())[0] for k in self.keys}
        merged = {k: sorted(sub[sub.key == k].phrase.unique()) for k in self.keys
                  if sub[sub.key == k].phrase.nunique() > 1}
        self.merged = merged
        self.eps = sorted(sub.episode_index.unique())
        zp = sub.pivot_table(index="key", columns="episode_index", values="z_row", aggfunc="mean")
        gp = sub.pivot_table(index="key", columns="episode_index", values="grip_row", aggfunc="mean")
        self.Z = zp.loc[self.keys, self.eps].values      # (K, E) mean z per episode
        self.G = gp.loc[self.keys, self.eps].values      # (K, E) mean grip per episode
        assert not np.isnan(self.Z).any() and not np.isnan(self.G).any(), \
            f"{task}: incomplete (key x episode) grid"
        self.echo = np.array([echo_count(self.raw_text[k], task) for k in self.keys], float)
        self.kidx = {k: i for i, k in enumerate(self.keys)}


def rank01(v):
    r = pd.Series(v).rank(method="average").values  # 1..n, ties averaged
    return (r - 1.0) / (len(v) - 1.0)


def winsorize(g):
    lo, hi = np.percentile(g, 5), np.percentile(g, 95)
    return np.clip(g, lo, hi)


def candidate_scores(z, g, echo, floor):
    """All candidate scores for one task's scored group (vectors over keys)."""
    med = float(np.median(z))
    gw = winsorize(g)
    z01, g01, gw01 = rank01(z), rank01(-g), rank01(-gw)
    gate = (z >= med) & (z >= floor)
    def gated(fine):
        return np.where(gate, 1000.0 + fine, z - 1000.0)
    out = {
        "C1_ens4f": z,
        "C2_relfloor_grip": gated(-g),
        "C3_relfloor_gripwins": gated(-gw),
        "C4a_rank_50_50": 0.5 * z01 + 0.5 * g01,
        "C4b_rank_25_75": 0.25 * z01 + 0.75 * g01,
    }
    for lam in LAMBDAS:
        tag = str(lam).replace(".", "")
        out[f"C5_echo_l{tag}"] = gated(gw01 - lam * echo)
        out[f"C5b_ens_echo_l{tag}"] = z01 - lam * echo
    return out, gate


CANDS = ["C1_ens4f", "C2_relfloor_grip", "C3_relfloor_gripwins", "C4a_rank_50_50",
         "C4b_rank_25_75", "C5_echo_l05", "C5_echo_l10", "C5b_ens_echo_l05",
         "C5b_ens_echo_l10"]


def blend(z, g, w):
    """rank-blend score for arbitrary w (weight on the gate-logit rank)."""
    return w * rank01(z) + (1.0 - w) * rank01(-g)


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def main():
    feats = load_features()
    ens = VerifierEnsemble("results/checkpoints/verifier_reward_ensemble_4f.json")
    feats["z_row"] = ens.member_logits(feats).mean(axis=1)
    feats["grip_row"] = [float(np.mean(v)) for v in feats.grip_err]

    states = {t: TaskState(t, sub) for t, sub in feats.groupby("task")}
    cov_sets = {t: {norm_cov(p) for p in sub.phrase.unique()} for t, sub in feats.groupby("task")}
    assert set(states) == set(NATIVE)

    # ---------------- 1. coverage matrix (all 139 frozen pairs) ----------------
    gz = json.load(open("results/analysis/gate_zero_pairs.json"))
    pairs = gz["pairs"]
    cov_rows, counts = [], {"native": {"both": 0, "one": 0, "none": 0},
                            "ood": {"both": 0, "one": 0, "none": 0}}
    for i, p in enumerate(pairs):
        t = p["task"]
        have = cov_sets.get(t, set())
        cb, cw = norm_cov(p["better"]) in have, norm_cov(p["worse"]) in have
        cov = "both" if cb and cw else ("one" if cb or cw else "none")
        cov_rows.append({"pair_index": i, "task": t, "native": t in NATIVE,
                         "covered": cov, "better_covered": cb, "worse_covered": cw,
                         "table": states[t].source if (t in states and cov != "none") else None,
                         "delta_pp": p["delta_pp"]})
        counts["native" if t in NATIVE else "ood"][cov] += 1

    print("=" * 70)
    print("COVERAGE (139 frozen pairs; features = fresh examfull tables only):")
    for grp in ("native", "ood"):
        c = counts[grp]
        print(f"  {grp:6s}: both={c['both']:3d}  one={c['one']:3d}  none={c['none']:3d}"
              f"  (total {sum(c.values())})")
    for t in NATIVE:
        s = states[t]
        print(f"  {t:34s} {len(s.keys):2d} phrases x {len(s.eps)} episode-clusters "
              f"({s.source}; merged keys: {len(s.merged)})")

    # ---------------- floor (fresh, same frozen rule as v2) ----------------
    per_key_z = {t: s.Z.mean(axis=1) for t, s in states.items()}
    nominal_z = {t: float(per_key_z[t][s.kidx[norm_key(NOMINAL[t])]])
                 for t, s in states.items()}
    bridge_tasks = [t for t in states if states[t].source == "examfull_bridge"]
    FLOOR = min(nominal_z[t] for t in bridge_tasks)
    print("\nPASSTHROUGH (nominal) ensemble logits per task (fresh):")
    for t in NATIVE:
        print(f"  {t:34s} z={nominal_z[t]:+.3f}"
              f"{'  (sim; excluded from floor derivation)' if states[t].source == 'examfull_egg_sim' else ''}")
    print(f"  FROZEN FLOOR = min over Bridge tasks = {FLOOR:+.3f}")

    # ---------------- point candidate scores ----------------
    point, gates = {}, {}
    for t, s in states.items():
        z, g = s.Z.mean(axis=1), s.G.mean(axis=1)
        point[t], gates[t] = candidate_scores(z, g, s.echo, FLOOR)

    def score_phrase(cand, task, text):
        s = states.get(task)
        if s is None:
            return None
        k = norm_key(text)
        if k not in s.kidx:
            return None
        i = s.kidx[k]
        base = point[task][cand][i]
        if cand.startswith("C5"):   # recompute penalty from the QUERY text (control-honest)
            lam = 0.5 if cand.endswith("l05") else 1.0
            stored = lam * s.echo[i]
            base = base + stored - lam * echo_count(text, task)
        return float(base)

    # ---------------- 2. gate-zero exam ----------------
    covered_pairs = [p for p, c in zip(pairs, cov_rows) if c["covered"] == "both"]
    assert all(c["task"] in NATIVE for c in cov_rows if c["covered"] == "both")
    gz_detail = {c: [] for c in CANDS}
    for p in covered_pairs:
        t = p["task"]
        for c in CANDS:
            sb, sw = score_phrase(c, t, p["better"]), score_phrase(c, t, p["worse"])
            gz_detail[c].append({
                "task": t, "better": p["better"], "worse": p["worse"],
                "delta_pp": p["delta_pp"], "sim": states[t].source == "examfull_egg_sim",
                "score_better": round(sb, 4), "score_worse": round(sw, 4),
                "correct": bool(sb > sw)})
    gz_summary = {}
    for c in CANDS:
        d = gz_detail[c]
        bridge = [r for r in d if not r["sim"]]
        sim = [r for r in d if r["sim"]]
        gz_summary[c] = {
            "native_correct": sum(r["correct"] for r in d), "native_n": len(d),
            "bridge_correct": sum(r["correct"] for r in bridge), "bridge_n": len(bridge),
            "sim_correct": sum(r["correct"] for r in sim), "sim_n": len(sim),
            "fails": [{"task": r["task"], "better": r["better"], "worse": r["worse"],
                       "delta_pp": r["delta_pp"],
                       "margin": round(r["score_better"] - r["score_worse"], 4),
                       "sim": r["sim"]} for r in d if not r["correct"]]}

    # ---------------- 3. regret exam vs the frozen success table ----------------
    succ_tab = pd.read_parquet("results/analysis/phrase_success_table.parquet")
    succ_tab["key"] = [norm_key(p) for p in succ_tab.phrase]

    def regret_from(scores_vec, kidxs, succ, top=1):
        order = np.argsort(-scores_vec[kidxs])
        chosen = succ[order[:top]]
        return float(succ.max() - chosen.max())

    def boot_regret(task, exam_keys, succ_mean, succ_se):
        """B draws: cluster-resample feature contexts (cluster = source episode;
        4 quartile frames per Bridge episode, 1 frame per sim context) for the
        score side. Two success treatments per draw:
          perturbed  success cells + N(0, layout-clustered se)  [v2 design; the
                     noise enters the regret statistic directly, so an ORACLE
                     ranker (rank = true point success) is scored too as the
                     mechanical noise floor]
          scorefix   success fixed at the point estimates -> CI reflects scorer
                     (feature) stability only."""
        s = states[task]
        kidxs = np.array([s.kidx[k] for k in exam_keys])
        n_e = len(s.eps)
        out = {c: {"top1": [], "top3": [], "top1_scorefix": [], "top3_scorefix": []}
               for c in CANDS}
        oracle = {"top1": [], "top3": []}
        orc_order = np.argsort(-succ_mean)
        for _ in range(B):
            eidx = RNG.choice(n_e, n_e)
            z, g = s.Z[:, eidx].mean(axis=1), s.G[:, eidx].mean(axis=1)
            sc, _ = candidate_scores(z, g, s.echo, FLOOR)
            succ_b = succ_mean + RNG.normal(0, succ_se)
            oracle["top1"].append(float(succ_b.max() - succ_b[orc_order[:1]].max()))
            oracle["top3"].append(float(succ_b.max() - succ_b[orc_order[:3]].max()))
            for c in CANDS:
                out[c]["top1"].append(regret_from(sc[c], kidxs, succ_b, 1))
                out[c]["top3"].append(regret_from(sc[c], kidxs, succ_b, 3))
                out[c]["top1_scorefix"].append(regret_from(sc[c], kidxs, succ_mean, 1))
                out[c]["top3_scorefix"].append(regret_from(sc[c], kidxs, succ_mean, 3))
        def ci(v):
            return {"mean": round(float(np.mean(v)), 4),
                    "p5": round(float(np.percentile(v, 5)), 4),
                    "p95": round(float(np.percentile(v, 95)), 4)}
        res = {c: {m: ci(v) for m, v in d.items()} for c, d in out.items()}
        res["ORACLE_noise_floor"] = {m: ci(v) for m, v in oracle.items()}
        return res

    frozen_regret, excluded_cells = {}, []
    for t in NATIVE:
        sub = succ_tab[succ_tab.task == t]
        excl = [{"task": t, "phrase": r.phrase, "succ": round(r.succ * 100, 1)}
                for r in sub.itertuples() if r.key not in states[t].kidx]
        excluded_cells += excl
        sub = sub[sub.key.isin(states[t].kidx)].drop_duplicates("key")
        keys = list(sub.key)
        sm = sub.set_index("key").loc[keys]
        succ, se = sm.succ.values * 100, sm.se.values * 100
        kidxs = np.array([states[t].kidx[k] for k in keys])
        res = {"n_covered_phrases": len(keys),
               "excluded_success_cells": excl,
               "phrases": [{"phrase": r.phrase, "succ": round(r.succ * 100, 1)}
                           for r in sm.itertuples()]}
        boot = boot_regret(t, keys, succ, se)
        for c in CANDS:
            res[c] = {"top1_point": round(regret_from(point[t][c], kidxs, succ, 1), 1),
                      "top3_point": round(regret_from(point[t][c], kidxs, succ, 3), 1),
                      "boot": boot[c]}
        res["ORACLE_noise_floor"] = boot["ORACLE_noise_floor"]
        frozen_regret[t] = res

    # ---------------- 3b. Spearman vs success (writeup support) ----------------
    spear = {c: {"per_task": {}, } for c in CANDS}
    pooled_pts = {c: {"score_r01": [], "succ_r01": []} for c in CANDS}
    for t in NATIVE:
        sub = succ_tab[(succ_tab.task == t) & succ_tab.key.isin(states[t].kidx)] \
            .drop_duplicates("key")
        keys = list(sub.key)
        succ = sub.set_index("key").loc[keys].succ.values * 100
        kidxs = np.array([states[t].kidx[k] for k in keys])
        for c in CANDS:
            sc = point[t][c][kidxs]
            spear[c]["per_task"][t] = {"rho": round(spearman(sc, succ), 3), "n": len(keys)}
            pooled_pts[c]["score_r01"] += list(rank01(sc))
            pooled_pts[c]["succ_r01"] += list(rank01(succ))
    for c in CANDS:
        per = spear[c]["per_task"]
        ns = np.array([v["n"] for v in per.values()], float)
        rs = np.array([v["rho"] for v in per.values()], float)
        spear[c]["mean_rho_n_weighted"] = round(float((ns * rs).sum() / ns.sum()), 3)
        spear[c]["pooled_within_task_ranks"] = round(
            spearman(pooled_pts[c]["score_r01"], pooled_pts[c]["succ_r01"]), 3)
        spear[c]["pooled_n"] = len(pooled_pts[c]["score_r01"])

    # ---------------- 3c. normalization control ----------------
    def perturb(text):
        yield "upper", text.upper()
        yield "period", text[:-1] if text.endswith(".") else text + "."
        yield "spaces", "  " + text.replace(" ", "  ") + "  "

    ctl_fail = {c: [] for c in CANDS}
    exam_texts = sorted({(p["task"], x) for p in covered_pairs for x in (p["better"], p["worse"])})
    for t, text in exam_texts:
        for c in CANDS:
            s0 = score_phrase(c, t, text)
            for tag, pt in perturb(text):
                s1 = score_phrase(c, t, pt)
                if s1 is None or s1 != s0:
                    ctl_fail[c].append({"task": t, "text": text, "perturb": tag,
                                        "orig": s0, "pert": s1})
    norm_control = {c: {"pass": not ctl_fail[c], "n_texts": len(exam_texts),
                        "n_perturbations": len(exam_texts) * 3, "fails": ctl_fail[c]}
                    for c in CANDS}

    # ---------------- 3d. leave-task-out transfer of the blend weight ----------
    # For each held-out task: choose w on the other 3 (fewest in-task sign fails,
    # then lowest max in-task top-1 regret, then lowest mean); check the 4th.
    task_blend = {}       # task -> w -> scores vector
    for t, s in states.items():
        z, g = s.Z.mean(axis=1), s.G.mean(axis=1)
        task_blend[t] = {w: blend(z, g, w) for w in W_GRID}
    task_pairs = {t: [p for p in covered_pairs if p["task"] == t] for t in NATIVE}
    task_exam = {}        # task -> (kidxs, succ)
    for t in NATIVE:
        sub = succ_tab[(succ_tab.task == t) & succ_tab.key.isin(states[t].kidx)] \
            .drop_duplicates("key")
        task_exam[t] = (np.array([states[t].kidx[k] for k in sub.key]),
                        sub.succ.values * 100)

    def w_metrics(t, w):
        sv = task_blend[t][w]
        fails = sum(1 for p in task_pairs[t]
                    if not (sv[states[t].kidx[norm_key(p["better"])]] >
                            sv[states[t].kidx[norm_key(p["worse"])]]))
        kidxs, succ = task_exam[t]
        return fails, regret_from(sv, kidxs, succ, 1)

    lto_folds = []
    for held in NATIVE:
        ins = [t for t in NATIVE if t != held]
        keyed = {}
        for w in W_GRID:
            ms = [w_metrics(t, w) for t in ins]
            keyed[w] = (sum(m[0] for m in ms), max(m[1] for m in ms),
                        float(np.mean([m[1] for m in ms])))
        best_key = min(keyed.values())
        tied_ws = [w for w in W_GRID if keyed[w] == best_key]
        w_star = tied_ws[0]
        best = (best_key, w_star)
        h_fails, h_reg = w_metrics(held, w_star)
        h_best = min(W_GRID, key=lambda w: (w_metrics(held, w)[0], w_metrics(held, w)[1]))
        hb_fails, hb_reg = w_metrics(held, h_best)
        lto_folds.append({
            "held_out": held, "chosen_w": w_star, "chosen_w_tie_set": tied_ws,
            "in_task_sign_fails": best[0][0], "in_task_max_top1": round(best[0][1], 1),
            "held_out_sign_fails_at_chosen": h_fails,
            "held_out_top1_at_chosen": round(h_reg, 1),
            "held_out_best_w": h_best, "held_out_sign_fails_at_best": hb_fails,
            "held_out_top1_at_best": round(hb_reg, 1),
            "holds": bool(h_reg <= REGRET_CAP_PP)})
    # full-grid per-task profile for the record
    w_profile = {t: {str(w): {"sign_fails": w_metrics(t, w)[0],
                              "top1_regret": round(w_metrics(t, w)[1], 1)}
                     for w in W_GRID} for t in NATIVE}

    # ---------------- 3e. fresh-vs-stored feature agreement (diagnostic) -------
    stored = load_stored_features()
    stored["z_row"] = ens.member_logits(stored).mean(axis=1)
    stored["grip_row"] = [float(np.mean(v)) for v in stored.grip_err]
    stored["key"] = [norm_key(p) for p in stored.phrase]
    fresh_vs_stored = {}
    feats["key"] = [norm_key(p) for p in feats.phrase]
    for t in NATIVE:
        fs = feats[feats.task == t].groupby("key")[["z_row", "grip_row"]].mean()
        st_ = stored[stored.task == t].groupby("key")[["z_row", "grip_row"]].mean()
        shared = sorted(set(fs.index) & set(st_.index))
        if not shared:
            fresh_vs_stored[t] = {"n_shared_phrases": 0}
            continue
        a, b = fs.loc[shared], st_.loc[shared]
        fresh_vs_stored[t] = {
            "n_shared_phrases": len(shared),
            "z_pearson": round(float(np.corrcoef(a.z_row, b.z_row)[0, 1]), 4)
            if len(shared) > 2 else None,
            "z_mean_abs_diff": round(float(np.abs(a.z_row - b.z_row).mean()), 4),
            "grip_mean_abs_diff": round(float(np.abs(a.grip_row - b.grip_row).mean()), 5),
        }

    # ---------------- 4. verdict + pre-registered selection ----------------
    # pairs no registered candidate gets right (drives the FAIL branch if any)
    unanimous_fails = []
    for i in range(len(covered_pairs)):
        if all(not gz_detail[c][i]["correct"] for c in CANDS):
            r = gz_detail[CANDS[0]][i]
            unanimous_fails.append({"task": r["task"], "better": r["better"],
                                    "worse": r["worse"], "delta_pp": r["delta_pp"],
                                    "sim": r["sim"]})

    def task_top1(c, t):
        return frozen_regret[t][c]["top1_point"]

    verdict_rows = []
    for c in CANDS:
        g = gz_summary[c]
        per_task = {t: task_top1(c, t) for t in NATIVE}
        p95s = {t: frozen_regret[t][c]["boot"]["top1"]["p95"] for t in NATIVE}
        p5s = {t: frozen_regret[t][c]["boot"]["top1"]["p5"] for t in NATIVE}
        sf95 = {t: frozen_regret[t][c]["boot"]["top1_scorefix"]["p95"] for t in NATIVE}
        maxreg = max(per_task.values())
        argmax_t = max(per_task, key=per_task.get)
        verdict_rows.append({
            "candidate": c,
            "gate_zero_native": f"{g['native_correct']}/{g['native_n']}",
            "gate_zero_bridge_only": f"{g['bridge_correct']}/{g['bridge_n']}",
            "passes_all_native_signs": g["native_correct"] == g["native_n"],
            "max_top1_regret": maxreg,
            "max_top1_regret_task": argmax_t,
            "top1_regret_per_task": per_task,
            "cap_pass_point": bool(maxreg <= REGRET_CAP_PP),
            # scorer-stability CI (success fixed at point estimates)
            "top1_scorefix_p95_per_task": sf95,
            "max_top1_scorefix_p95": max(sf95.values()),
            "ci_cleared_scorefix_6pp": bool(max(sf95.values()) <= REGRET_CAP_PP),
            # perturbed-success CI (v2 design; compare ORACLE noise floor)
            "top1_perturbed_p95_per_task": p95s,
            "max_top1_perturbed_p95": max(p95s.values()),
            "ci_lowerbound_violates_cap": bool(max(p5s.values()) > REGRET_CAP_PP),
            "norm_control_pass": norm_control[c]["pass"],
            "spearman_pooled": spear[c]["pooled_within_task_ranks"],
        })

    passers = [r for r in verdict_rows
               if r["passes_all_native_signs"] and r["norm_control_pass"]]
    winner, winner_note, tie = None, None, []
    if passers:
        best_reg = min(r["max_top1_regret"] for r in passers)
        tie = [r for r in passers if r["max_top1_regret"] == best_reg]
        if len(tie) > 1:
            winner_note = (f"{len(tie)} sign-passers tie on max top-1 regret "
                           f"({best_reg}pp); tie-break = LTO blend-weight transfer")
            blends = [r for r in tie if r["candidate"].startswith("C4")]
            if blends:
                # LTO decides among the blend family: pick the pre-registered w
                # chosen by every fold if unanimous
                ws = {f["chosen_w"] for f in lto_folds}
                if len(ws) == 1 and list(ws)[0] in W_PREREG:
                    pick = "C4a_rank_50_50" if list(ws)[0] == 0.5 else "C4b_rank_25_75"
                    cand = [r for r in blends if r["candidate"] == pick]
                    winner = cand[0] if cand else None
        if winner is None and len(tie) == 1:
            winner = tie[0]
        if winner is None and len(tie) > 1:
            # LTO cannot separate non-blend ties; fall back to reporting the tie,
            # ordered by scorer-stability CI then pooled Spearman (NOT pre-registered).
            winner = sorted(tie, key=lambda r: (r["max_top1_scorefix_p95"],
                                                -r["spearman_pooled"]))[0]
            winner_note = (winner_note or "") + \
                "; residual tie ordered by scorer-stability p95 then pooled " \
                "Spearman (reported, not pre-registered)"
    # certification: point regret under cap, scorer-stability CI under cap, and
    # the perturbed-success CI lower bound must not prove a violation
    certified = bool(winner and winner["cap_pass_point"]
                     and winner["ci_cleared_scorefix_6pp"]
                     and not winner["ci_lowerbound_violates_cap"])
    if not passers:
        statement = ("NO WINNER: no candidate passes all 68 covered native "
                     "gate-zero signs + normalization control -> FAIL branch of "
                     "REWARD_PLAN.md Step 3 (escalate: coarse-gate+echo only / "
                     "guardrailed rollout reward / writeup-first). Best sign "
                     "accuracy: " + ", ".join(
                         f"{r['candidate']} {r['gate_zero_native']}"
                         for r in sorted(verdict_rows,
                                         key=lambda r: -int(r['gate_zero_native']
                                                            .split('/')[0]))[:2]) + ".")
    elif certified:
        statement = (f"WINNER: {winner['candidate']} -- passes 68/68 native "
                     f"gate-zero signs, max top-1 regret "
                     f"{winner['max_top1_regret']}pp (scorer-stability p95 "
                     f"{winner['max_top1_scorefix_p95']}pp <= {REGRET_CAP_PP}pp "
                     f"cap), normalization control clean."
                     + (f" [{winner_note}]" if winner_note else ""))
    else:
        statement = (f"NO CERTIFIED WINNER: best sign-passer "
                     f"{winner['candidate']} has max top-1 regret "
                     f"{winner['max_top1_regret']}pp but fails CI clearance vs "
                     f"the {REGRET_CAP_PP}pp cap (scorer-stability p95 "
                     f"{winner['max_top1_scorefix_p95']}pp)."
                     + (f" [{winner_note}]" if winner_note else ""))

    ranking = sorted(verdict_rows, key=lambda r: (
        not (r["passes_all_native_signs"] and r["norm_control_pass"]),
        -int(r["gate_zero_native"].split("/")[0]),
        r["max_top1_regret"], r["max_top1_scorefix_p95"], -r["spearman_pooled"]))

    report = {
        "created": "2026-07-18",
        "script": "results/analysis/reward_bakeoff_v3.py",
        "feature_source": {
            "bridge": ["examfull_verbose_bridge.parquet", "examfull_l2_bridge.parquet",
                       "contexts = 200 Bridge quartile contexts (multit_astar.parquet)"],
            "eggplant": ["examfull_verbose_egg.parquet", "examfull_l2_egg.parquet",
                         "contexts = 60 sim contexts (sim_contexts_eggplant.parquet); "
                         "learned fine readouts carry the recorded sim-inversion caveat"],
            "note": "ALL features fresh from one same-GPU pass over the 36 powered "
                    "exam phrases; stored study tables used only for the "
                    "fresh-vs-stored agreement diagnostic"},
        "frozen_decisions": {
            "floor": {"value": round(FLOOR, 4),
                      "nominal_logits": {t: round(v, 4) for t, v in nominal_z.items()},
                      "rule": "min over Bridge-context native tasks of the nominal-"
                              "passthrough ensemble logit (sim eggplant excluded from "
                              "derivation, floor applies everywhere; no margin)"},
            "group": "within-task scored group = the task's EXAM phrase set "
                     "(9/11/8/8) -- CHANGED from v2's 35-phrase study group as a "
                     "documented consequence of sourcing all features from the "
                     "examfull tables; candidate formulas unchanged",
            "c5_penalty": "fine = rank01(-grip_winsorized) - lambda*|distinct lexicon "
                          "tokens in candidate and not in nominal|; "
                          "C5b = rank01(ensemble logit) - lambda*echo; lambda in {0.5, 1.0}",
            "bootstrap": f"B={B}; score side cluster-resamples source episodes "
                         "(4 quartile frames per Bridge episode, 1 frame per sim "
                         "context); success cells perturbed N(0, layout-clustered "
                         "se); FLOOR and echo frozen across draws",
            "regret_cap_pp": REGRET_CAP_PP},
        "coverage_counts": counts,
        "coverage_pairs": cov_rows,
        "per_task_tables": {t: {"source": s.source, "n_phrases": len(s.keys),
                                "n_episode_clusters": len(s.eps),
                                "merged_keys": s.merged} for t, s in states.items()},
        "gate_zero": {"n_covered_pairs": len(covered_pairs),
                      "summary": gz_summary,
                      "unanimous_fail_pairs": unanimous_fails,
                      "detail": gz_detail},
        "frozen_regret": frozen_regret,
        "oracle_noise_floor_note": "ORACLE_noise_floor per task = regret CI of a "
            "ranker that ranks by the TRUE point success values, under the same "
            "success perturbation; any candidate p95 must be read against it -- "
            "the perturbed-success p95 cannot certify the 6pp cap when the floor "
            "itself exceeds the cap (success-measurement noise, not scorer error)",
        "excluded_success_cells": excluded_cells,
        "spearman_vs_success": spear,
        "normalization_control": {c: {k: v for k, v in norm_control[c].items()}
                                  for c in CANDS},
        "lto_blend_weight": {"grid": W_GRID, "preregistered_w": W_PREREG,
                             "selection_on_3_tasks": "fewest in-task sign fails -> "
                             "lowest max in-task top-1 regret -> lowest mean",
                             "folds": lto_folds, "per_task_w_profile": w_profile},
        "fresh_vs_stored_diagnostic": fresh_vs_stored,
        "verdict_table": verdict_rows,
        "selection_rule": "pass ALL covered native gate-zero signs AND norm control "
                          "-> lowest max top-1 regret, CI-cleared vs the 6pp cap "
                          "(point <= 6pp; scorer-stability p95 <= 6pp; perturbed-"
                          "success CI lower bound must not exceed 6pp) -> tie-break "
                          "leave-task-out transfer of the blend weight",
        "passers": [r["candidate"] for r in passers],
        "ranking": [r["candidate"] for r in ranking],
        "winner": winner["candidate"] if (passers and winner) else None,
        "certified": certified,
        "winner_statement": statement,
    }

    def np_safe(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serializable: {type(o)}")

    json.dump(report, open("results/analysis/reward_bakeoff_v3.json", "w"),
              indent=1, default=np_safe)

    # ---------------- prints ----------------
    n_sim = sum(1 for p in covered_pairs if states[p["task"]].source == "examfull_egg_sim")
    print("\n" + "=" * 70)
    print(f"GATE-ZERO over {len(covered_pairs)} covered native pairs "
          f"({len(covered_pairs) - n_sim} bridge, {n_sim} sim); OOD coverable: "
          f"{counts['ood']['both']}")
    hdr = (f"{'candidate':22s} {'GZnat':>6s} {'GZbr':>6s} {'maxT1':>6s} {'sf-p95':>6s} "
           f"{'CIsf<=6':>7s} {'norm':>5s} {'rho':>6s}")
    print(hdr)
    for r in verdict_rows:
        print(f"{r['candidate']:22s} {r['gate_zero_native']:>6s} "
              f"{r['gate_zero_bridge_only']:>6s} {r['max_top1_regret']:>6.1f} "
              f"{r['max_top1_scorefix_p95']:>6.1f} "
              f"{str(r['ci_cleared_scorefix_6pp']):>7s} "
              f"{str(r['norm_control_pass']):>5s} {r['spearman_pooled']:>6.3f}")
    print("\nPer-task top-1 regret point/scorer-stability-p95 "
          "(ORACLE = perturbed-success noise floor p95):")
    for t in NATIVE:
        cells = "  ".join(f"{c.split('_')[0]}={frozen_regret[t][c]['top1_point']:.1f}/"
                          f"{frozen_regret[t][c]['boot']['top1_scorefix']['p95']:.1f}"
                          for c in CANDS)
        orc = frozen_regret[t]["ORACLE_noise_floor"]["top1"]["p95"]
        print(f"  {t:34s} ORACLE={orc:.1f}  {cells}")
    if unanimous_fails:
        print("\nPairs failed by ALL 9 candidates:")
        for u in unanimous_fails:
            print(f"  {u['task']} (d={u['delta_pp']}pp, sim={u['sim']}): "
                  f"{u['better']!r} > {u['worse']!r}")
    print("\nLTO blend-weight folds:")
    for f in lto_folds:
        print(f"  held={f['held_out']:34s} w*={f['chosen_w']:.3f} "
              f"held-out top1@w*={f['held_out_top1_at_chosen']:.1f}pp "
              f"(best-w {f['held_out_best_w']:.3f} -> "
              f"{f['held_out_top1_at_best']:.1f}pp) holds={f['holds']}")
    print("\n" + "=" * 70)
    print(statement)
    print("Artifacts: results/analysis/reward_bakeoff_v3.json")


if __name__ == "__main__":
    main()
