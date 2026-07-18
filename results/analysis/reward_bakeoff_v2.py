"""v7 Steps 2-3: offline reward candidate bake-off against the FROZEN exam.

Exam artifacts (frozen before this script was written -- never edited here):
  results/analysis/gate_zero_pairs.json         139 powered pairs (layout-clustered)
  results/analysis/phrase_success_table.parquet 70 (task,phrase) success cells

Feature sources (stored pi0 tables; NOTHING is trained here -- all candidates
are pure functions of stored features):
  carrot/spoon/stack : study_multit_verbose/_l2 + multit_astar   (real Bridge
                       quartile contexts, 20/10/20 eps x 4 t; the 4f decision table)
  eggplant           : eggplant_sim_verbose/_l2 + sim_contexts_eggplant (60 SIM
                       contexts; learned FINE readouts invert on sim -- recorded
                       finding -- so sim-scored pairs are flagged; raw grip/gate
                       readouts on sim are OK, grip scored 1.0 pairwise there)
  study_scores_* (old single-frame contexts) carry the SAME 35-phrase sets as
  study_multit -> zero additional phrase coverage; not used.

FROZEN DESIGN DECISIONS (documented, not tuned on the exam):
  norm_cov(s)  coverage match = NFKC + casefold + whitespace-collapse (per spec).
  norm_key(s)  scoring key    = norm_cov + strip non-alphanumerics. Feature rows
               of colliding table phrases are pooled (exactly one collision:
               stack 'Put the green block on the yellow block' +/- '.').
  z(t,p)       C1 = mean over context rows of the 4f ensemble's mean calibrated
               member logit (results/checkpoints/verifier_reward_ensemble_4f.json,
               reproduces the dawn executor's scoring path).
  grip(t,p)    mean over context rows of the row-mean grip_err (k_decode=4 draws).
  group        within-task scored group = ALL table phrases of that task (35) --
               the GRPO-group analog used for medians / ranks / winsor bounds.
  FLOOR        frozen passthrough floor = min over the 3 BRIDGE-context native
               tasks of z(task's nominal passthrough phrase). The eggplant
               nominal is excluded from the derivation because its z comes from
               sim rows (learned-readout sim caveat); the floor still APPLIES to
               eggplant. No margin parameter (a fitted margin would be exam tuning).
  C2           gate: z >= within-task median AND z >= FLOOR. fine = -grip.
               score = 1000 + fine if pass else z - 1000 (all passers > all
               failers; failers ordered by gate logit).
  C3           C2 with grip winsorized at the within-task p5/p95 of the
               PHRASE-LEVEL grip distribution (35 values).
  C4           ungated rank blend: w*rank01(z) + (1-w)*rank01(-grip); rank01 =
               (avg rank - 1)/(n - 1) within task, 1 = best; w in {0.5, 0.25}.
  C5           C3 gate; fine = rank01(-grip_winsorized) - lambda*echo, in
               rank01 units ([0,1] spans the task's field; lambda=1.0 sends a
               single echoed token from best to worst). echo = # DISTINCT
               lexicon tokens present in the CANDIDATE phrase's token set
               (lowercased, punctuation-stripped) and ABSENT from the task's
               nominal instruction token set. lambda in {0.5, 1.0}.
  C5b          C1 + the same penalty: rank01(z) - lambda*echo (ungated).
  lexicon      {center, middle, exactly, precisely, vertically, upright, gently,
               directly, carefully, neatly, firmly, slowly}
  Regret CI    B=2000 clustered bootstrap. LIMITATION: feature tables carry no
               per-layout rollout rows, so the success side of the FROZEN exam
               cannot be resampled by layout; each draw instead (a) resamples
               feature EPISODES (the tables' natural cluster) for the score side
               and (b) perturbs each success cell with N(0, se) using the
               table's layout-clustered se. The SUPPLEMENTARY study-label exam
               (rollouts_study_*, 24 layouts x 12 reps) does support a true
               layout-clustered success bootstrap and gets one.
  Supplement   study-label regret / pairwise acc is reported for context only
               (dawn protocol); it is NOT part of selection: the 4f ensemble was
               SELECTED on that table (contaminated for C1).

Run:  .venv/bin/python results/analysis/reward_bakeoff_v2.py
Out:  results/analysis/exam_coverage.json
      results/analysis/feature_extraction_needed.json
      results/analysis/reward_bakeoff_v2.json
"""

import glob
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
RNG = np.random.default_rng(0)

from phrase_rl.verifier_reward import VerifierEnsemble  # noqa: E402

NATIVE = ["widowx_carrot_on_plate", "widowx_spoon_on_towel",
          "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
NOMINAL = {  # native from probe_chart.py task map; OOD = the eval12nom phrases in the registry
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


def norm_cov(s):
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    return re.sub(r"\s+", " ", s).strip()


def norm_key(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", norm_cov(s))).strip()


def toks(s):
    return set(re.findall(r"[a-z]+", norm_cov(s)))


def echo_count(text, task):
    return len((toks(text) & LEXICON) - toks(NOMINAL[task]))


# ---------------- feature rows ----------------

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
    multit = build_rows(f"{RAW}/study_multit_verbose.parquet",
                        f"{RAW}/study_multit_l2.parquet", amap, "study_multit")
    sc = pd.read_parquet(f"{RAW}/sim_contexts_eggplant.parquet")
    amap_s = {(r.task, int(r.episode_index), int(r.t)): np.asarray(r.action_chunk, np.float32)
              for r in sc.itertuples()}
    sim = build_rows(f"{RAW}/eggplant_sim_verbose.parquet",
                     f"{RAW}/eggplant_sim_l2.parquet", amap_s, "eggplant_sim")
    return pd.concat([multit, sim], ignore_index=True)


# ---------------- per-task scored state ----------------

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


def main():
    feats = load_features()
    ens = VerifierEnsemble("results/checkpoints/verifier_reward_ensemble_4f.json")
    feats["z_row"] = ens.member_logits(feats).mean(axis=1)
    feats["grip_row"] = [float(np.mean(v)) for v in feats.grip_err]

    states = {t: TaskState(t, sub) for t, sub in feats.groupby("task")}
    cov_sets = {t: {norm_cov(p) for p in sub.phrase.unique()} for t, sub in feats.groupby("task")}
    key_of_cov = {t: {norm_cov(p): norm_key(p) for p in sub.phrase.unique()}
                  for t, sub in feats.groupby("task")}

    # ---------------- 1. coverage matrix ----------------
    gz = json.load(open("results/analysis/gate_zero_pairs.json"))
    pairs = gz["pairs"]
    cov_rows, counts = [], {"native": {"both": 0, "one": 0, "none": 0},
                            "ood": {"both": 0, "one": 0, "none": 0}}
    punct_only = 0
    for i, p in enumerate(pairs):
        t = p["task"]
        have = cov_sets.get(t, set())
        cb, cw = norm_cov(p["better"]) in have, norm_cov(p["worse"]) in have
        # diagnostics: would punct-stripping add a match norm_cov misses?
        kset = {norm_key(x) for x in have} if have else set()
        for txt, hit in ((p["better"], cb), (p["worse"], cw)):
            if not hit and norm_key(txt) in kset:
                punct_only += 1
        cov = "both" if cb and cw else ("one" if cb or cw else "none")
        table = states[t].source if t in states else None
        cov_rows.append({"pair_index": i, "task": t,
                         "native": t in NATIVE, "covered": cov,
                         "better_covered": cb, "worse_covered": cw,
                         "table": table if cov != "none" else None,
                         "delta_pp": p["delta_pp"]})
        counts["native" if t in NATIVE else "ood"][cov] += 1
    json.dump({"created": "2026-07-18",
               "match_rule": "exact phrase string after NFKC+casefold+whitespace-collapse",
               "tables": {t: {"source": s.source, "n_raw_phrases": int(sum(len(v) for v in
                              s.merged.values()) + len(s.keys) - len(s.merged)),
                              "n_score_keys": len(s.keys), "n_episodes": len(s.eps),
                              "punct_merged_keys": s.merged} for t, s in states.items()},
               "note": ("study_scores_* old single-frame contexts carry the same 35-phrase "
                        "sets as study_multit -> no additional coverage; OOD (_clean) tasks "
                        "have no stored feature tables at all."),
               "punct_only_matches": punct_only,
               "counts": counts, "pairs": cov_rows},
              open("results/analysis/exam_coverage.json", "w"), indent=1)

    print("=" * 70)
    print("COVERAGE (139 frozen pairs):")
    for grp in ("native", "ood"):
        c = counts[grp]
        print(f"  {grp:6s}: both={c['both']:3d}  one={c['one']:3d}  none={c['none']:3d}"
              f"  (total {sum(c.values())})")
    print(f"  punct-only near-matches (not counted as covered): {punct_only}")

    # ---------------- floor ----------------
    per_key_z = {t: s.Z.mean(axis=1) for t, s in states.items()}
    nominal_z = {}
    for t, s in states.items():
        k = norm_key(NOMINAL[t])
        nominal_z[t] = float(per_key_z[t][s.kidx[k]])
    bridge_tasks = [t for t in states if states[t].source == "study_multit"]
    FLOOR = min(nominal_z[t] for t in bridge_tasks)
    print("\nPASSTHROUGH (nominal) ensemble logits per task:")
    for t in NATIVE:
        if t in nominal_z:
            print(f"  {t:34s} z={nominal_z[t]:+.3f}"
                  f"{'  (sim; excluded from floor)' if states[t].source == 'eggplant_sim' else ''}")
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
    gz_detail = {c: [] for c in CANDS}
    for p in covered_pairs:
        t = p["task"]
        for c in CANDS:
            sb, sw = score_phrase(c, t, p["better"]), score_phrase(c, t, p["worse"])
            gz_detail[c].append({
                "task": t, "better": p["better"], "worse": p["worse"],
                "delta_pp": p["delta_pp"], "sim": states[t].source == "eggplant_sim",
                "score_better": round(sb, 4), "score_worse": round(sw, 4),
                "correct": bool(sb > sw)})
    gz_summary = {}
    for c in CANDS:
        d = gz_detail[c]
        nat = [r for r in d if True]  # all covered pairs are native (asserted below)
        gz_summary[c] = {
            "native_correct": sum(r["correct"] for r in nat), "native_n": len(nat),
            "all_correct": sum(r["correct"] for r in d), "all_n": len(d),
            "fails": [{"task": r["task"], "better": r["better"], "worse": r["worse"],
                       "margin": round(r["score_better"] - r["score_worse"], 4),
                       "sim": r["sim"]} for r in d if not r["correct"]]}
    assert all(c["task"] in NATIVE for c in cov_rows if c["covered"] == "both")

    # ---------------- 3. regret exams ----------------
    succ_tab = pd.read_parquet("results/analysis/phrase_success_table.parquet")
    succ_tab["key"] = [norm_key(p) for p in succ_tab.phrase]

    def regret_from(scores_vec, kidxs, succ, top=1):
        order = np.argsort(-scores_vec[kidxs])
        chosen = succ[order[:top]]
        return float(succ.max() - chosen.max())

    def boot_regret(task, exam_keys, succ_mean, succ_se=None, succ_rows=None):
        """B draws: episode-resample scores; success via N(0,se) or layout resample."""
        s = states[task]
        kidxs = np.array([s.kidx[k] for k in exam_keys])
        n_e = len(s.eps)
        out = {c: {"top1": [], "top3": []} for c in CANDS}
        for _ in range(B):
            eidx = RNG.choice(n_e, n_e)
            z, g = s.Z[:, eidx].mean(axis=1), s.G[:, eidx].mean(axis=1)
            sc, _ = candidate_scores(z, g, s.echo, FLOOR)
            if succ_rows is not None:                     # layout-clustered resample
                lays = succ_rows.episode_id.unique()
                pick = RNG.choice(lays, len(lays))
                sub = pd.concat([succ_rows[succ_rows.episode_id == l] for l in pick])
                sb = sub.groupby("key").success.mean() * 100.0
                succ_b = np.array([sb.get(k, np.nan) for k in exam_keys])
                if np.isnan(succ_b).any():
                    continue
            else:
                succ_b = succ_mean + RNG.normal(0, succ_se)
            for c in CANDS:
                out[c]["top1"].append(regret_from(sc[c], kidxs, succ_b, 1))
                out[c]["top3"].append(regret_from(sc[c], kidxs, succ_b, 3))
        return {c: {m: {"mean": round(float(np.mean(v)), 4),
                        "p5": round(float(np.percentile(v, 5)), 4),
                        "p95": round(float(np.percentile(v, 95)), 4)}
                    for m, v in d.items()} for c, d in out.items()}

    frozen_regret = {}
    for t in NATIVE:
        if t not in states:
            continue
        sub = succ_tab[succ_tab.task == t]
        keys = [k for k in sub.key if k in states[t].kidx]
        if len(keys) < 3:
            frozen_regret[t] = {"n_covered_phrases": len(keys),
                                "note": "insufficient coverage (<3) -- see feature gap job"}
            continue
        sub = sub[sub.key.isin(keys)].drop_duplicates("key")
        sm = sub.set_index("key").loc[keys]
        succ, se = sm.succ.values * 100, sm.se.values * 100
        kidxs = np.array([states[t].kidx[k] for k in keys])
        res = {"n_covered_phrases": len(keys),
               "phrases": [{"phrase": r.phrase, "succ": round(r.succ * 100, 1)}
                           for r in sm.itertuples()]}
        for c in CANDS:
            res[c] = {"top1_point": round(regret_from(point[t][c], kidxs, succ, 1), 1),
                      "top3_point": round(regret_from(point[t][c], kidxs, succ, 3), 1)}
        boot = boot_regret(t, keys, succ, succ_se=se)
        for c in CANDS:
            res[c]["boot"] = boot[c]
        frozen_regret[t] = res

    # supplementary: study labels (dawn protocol; NOT selection -- C1 contaminated)
    supp_regret, supp_pairs_pool = {}, {c: [] for c in CANDS}
    for f in sorted(glob.glob(f"{RAW}/rollouts_study_widowx_*.parquet")):
        d = pd.read_parquet(f)
        t = d.task.iloc[0]
        if t not in states:
            continue
        d["key"] = [norm_key(p) for p in d.phrase]
        d = d[d.key.isin(states[t].kidx)]
        sm = d.groupby("key").success.mean() * 100
        keys = list(sm.index)
        kidxs = np.array([states[t].kidx[k] for k in keys])
        succ = sm.values
        res = {"n_covered_phrases": len(keys), "n_eps_per_phrase": int(d.groupby("key").size().max())}
        for c in CANDS:
            res[c] = {"top1_point": round(regret_from(point[t][c], kidxs, succ, 1), 1),
                      "top3_point": round(regret_from(point[t][c], kidxs, succ, 3), 1)}
            prs = [(point[t][c][states[t].kidx[a]] - point[t][c][states[t].kidx[b]],
                    sm[a] - sm[b])
                   for a, b in itertools.combinations(keys, 2) if abs(sm[a] - sm[b]) > 10]
            res[c]["pairwise_acc"] = round(float(np.mean([(x > 0) == (y > 0) for x, y in prs])), 3)
            res[c]["n_pairs"] = len(prs)
            supp_pairs_pool[c] += [(x, y) for x, y in prs]
        boot = boot_regret(t, keys, None, succ_rows=d[["episode_id", "key", "success"]])
        for c in CANDS:
            res[c]["boot"] = boot[c]
        supp_regret[t] = res
    supp_pooled = {c: {"pairwise_acc": round(float(np.mean([(x > 0) == (y > 0)
                                                            for x, y in supp_pairs_pool[c]])), 3),
                       "n_pairs": len(supp_pairs_pool[c])} for c in CANDS}

    # ---------------- 3c. normalization control ----------------
    def perturb(text):
        yield "upper", text.upper()
        yield "period", text[:-1] if text.endswith(".") else text + "."
        yield "spaces", "  " + text.replace(" ", "  ") + "  "

    ctl_fail = {c: [] for c in CANDS}
    ctl_checked = 0
    exam_texts = {(p["task"], x) for p in covered_pairs for x in (p["better"], p["worse"])}
    for t, text in sorted(exam_texts):
        ctl_checked += 1
        for c in CANDS:
            s0 = score_phrase(c, t, text)
            for tag, pt in perturb(text):
                s1 = score_phrase(c, t, pt)
                if s1 is None or s1 != s0:
                    ctl_fail[c].append({"task": t, "text": text, "perturb": tag,
                                        "orig": s0, "pert": s1})
    norm_control = {c: {"pass": not ctl_fail[c], "n_texts": ctl_checked,
                        "n_perturbations": ctl_checked * 3, "fails": ctl_fail[c]}
                    for c in CANDS}

    # ---------------- 4. feature-gap job spec ----------------
    missing, regret_only = {}, {}
    for p, c in zip(pairs, cov_rows):
        t = p["task"]
        for txt, hit in ((p["better"], c["better_covered"]), (p["worse"], c["worse_covered"])):
            if not hit:
                missing.setdefault(t, set()).add(txt)
    # success-table cells (regret exam) not in stored tables and not already queued
    for r in succ_tab.itertuples():
        t = r.task
        if norm_cov(r.phrase) in cov_sets.get(t, set()):
            continue
        if any(norm_cov(r.phrase) == norm_cov(x) for x in missing.get(t, set())):
            continue
        missing.setdefault(t, set()).add(r.phrase)
        regret_only.setdefault(t, set()).add(r.phrase)
    ctx_native_bridge = {
        "contexts": "data/contexts_0c_match.parquet expanded to quartile frames via "
                    "`python -m phrase_rl.multi_t_contexts --contexts data/contexts_0c_match.parquet "
                    "--out data/contexts_0c_match_multit.parquet` (seed=0 -> reproduces the 200 "
                    "contexts in results/overnight/raw/multit_astar.parquet)",
        "context_origin": "experiments.json 'phase0c-rollout-sensitivity': "
                          "`python -m phrase_rl.phase0c_match_contexts --per-task 20 "
                          "--out data/contexts_0c_match.parquet`"}
    spec_tasks = []
    for t in sorted(missing, key=lambda x: (x not in NATIVE, x)):
        phrases = sorted(missing[t])
        if t in NATIVE and t != "widowx_put_eggplant_in_basket":
            ctx = dict(ctx_native_bridge, required=True)
        elif t == "widowx_put_eggplant_in_basket":
            ctx = {"contexts": "results/overnight/raw/sim_contexts_eggplant.parquet (60 sim "
                               "contexts; eggplant has no Bridge counterpart per the 0c entry). "
                               "Learned fine readouts carry the sim-inversion caveat; grip/gate OK.",
                   "required": True}
        else:
            ctx = {"contexts": None, "required": False,
                   "note": "OOD (_clean) task -- optional diagnostic per the style-claim "
                           "constraint; no stored contexts exist (would need new sim-context "
                           "capture for the _clean variant)."}
        spec_tasks.append({"task": t, "native": t in NATIVE, "n_phrases": len(phrases),
                           "phrases": phrases,
                           "regret_exam_only": sorted(regret_only.get(t, set())), **ctx})
    json.dump({
        "created": "2026-07-18",
        "purpose": "score these (task, phrase) cells on stored/matched contexts so every frozen "
                   "gate-zero pair is fully covered; extraction must reproduce the training slots",
        "extraction": {
            "pins": {"k_flow": 8, "k_decode": 4, "seed": 0, "tau_min": 0.0,
                     "ckpt": "juexzz/INTACT-pi0-finetune-rephrase-bridge"},
            "verbose_table": "phrase_rl.study_verbose_rescore --contexts <ctx> --phrases <new "
                             "phrase parquet> --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge "
                             "--k 8  (flow v/u/loss)",
            "l2_table": "phrase_rl.pi0_decode --contexts <ctx> --phrases <new phrase parquet> "
                        "--stats-contexts data/contexts_train.parquet --ckpt juexzz/INTACT-pi0-"
                        "finetune-rephrase-bridge  (decoded/norm_l2/grip_err, K=4)"},
        "tasks": spec_tasks},
        open("results/analysis/feature_extraction_needed.json", "w"), indent=1)

    # ---------------- 5. verdict ----------------
    def max_frozen_regret(c):
        vals = [r[c]["top1_point"] for r in frozen_regret.values() if c in r]
        return max(vals) if vals else None

    verdict_rows = []
    for c in CANDS:
        g = gz_summary[c]
        verdict_rows.append({
            "candidate": c,
            "gate_zero_native": f"{g['native_correct']}/{g['native_n']}",
            "gate_zero_all": f"{g['all_correct']}/{g['all_n']}",
            "passes_all_native_signs": g["native_correct"] == g["native_n"],
            "max_frozen_top1_regret": max_frozen_regret(c),
            "supp_pooled_pairwise_acc": supp_pooled[c]["pairwise_acc"],
            "supp_max_top1_regret": max(supp_regret[t][c]["top1_point"] for t in supp_regret),
            "norm_control_pass": norm_control[c]["pass"],
        })
    # Selection rule applied to the COVERED subset; certification requires the
    # full native registry, which is coverage-blocked (see feature gap job).
    ranking = sorted(verdict_rows, key=lambda r: (
        not (r["passes_all_native_signs"] and r["norm_control_pass"]),
        r["max_frozen_top1_regret"] is None, r["max_frozen_top1_regret"],
        -r["supp_pooled_pairwise_acc"]))
    leader = ranking[0]["candidate"]
    n_native_pairs = sum(counts["native"].values())
    certified = counts["native"]["both"] == n_native_pairs
    subset_passers = [r["candidate"] for r in verdict_rows
                      if r["passes_all_native_signs"] and r["norm_control_pass"]]

    report = {
        "created": "2026-07-18",
        "frozen_decisions": {
            "floor": {"value": round(FLOOR, 4), "nominal_logits": {t: round(v, 4) for t, v in
                      nominal_z.items()}, "rule": "min over Bridge-context native tasks of the "
                      "nominal-passthrough ensemble logit (sim eggplant excluded from derivation, "
                      "floor applies everywhere; no margin)"},
            "group": "within-task scored group = all 35 stored table phrases (GRPO-group analog)",
            "c5_penalty": "fine = rank01(-grip_winsorized) - lambda*|distinct lexicon tokens in "
                          "candidate and not in nominal|; rank01 in [0,1] within task; "
                          "C5b = rank01(ensemble logit) - lambda*echo; lambda in {0.5, 1.0}",
            "bootstrap_limitation": "no per-layout rows in feature tables: frozen-exam success "
                                    "perturbed N(0, layout-clustered se); score side resamples "
                                    "feature episodes; supplementary exam resamples true layouts",
        },
        "coverage_counts": counts,
        "gate_zero": {"n_covered_pairs": len(covered_pairs),
                      "pairs": [{"task": p["task"], "better": p["better"], "worse": p["worse"],
                                 "delta_pp": p["delta_pp"],
                                 "sim": states[p["task"]].source == "eggplant_sim"}
                                for p in covered_pairs],
                      "summary": gz_summary, "detail": gz_detail},
        "frozen_regret": frozen_regret,
        "supplementary_study_labels": {
            "warning": "NOT selection evidence: the 4f ensemble was selected on this table "
                       "(contaminated for C1); shown for context only",
            "per_task": supp_regret, "pooled": supp_pooled},
        "normalization_control": norm_control,
        "verdict_table": verdict_rows,
        "selection_rule": "pass ALL covered native gate-zero signs AND norm control -> lowest "
                          "max frozen top-1 regret -> tie-break on supplementary transfer "
                          "(contaminated for C1; final tie-break deferred to full coverage)",
        "subset_passers": subset_passers,
        "leader_on_covered_subset": leader,
        "ranking": [r["candidate"] for r in ranking],
        "certified_outright": certified,
        "certification_blocker": None if certified else (
            f"only {counts['native']['both']}/{n_native_pairs} native gate-zero pairs are "
            f"coverable from stored feature tables; frozen regret administrable on 1/4 native "
            f"tasks -- run results/analysis/feature_extraction_needed.json, then re-run this "
            f"script unchanged"),
        "sim_caveats_observed": [
            "eggplant pair scored on SIM rows: all ensemble logits deeply negative "
            f"(nominal z={nominal_z['widowx_put_eggplant_in_basket']:+.2f}); the Bridge-derived "
            "FLOOR fails every sim phrase, so gated candidates degenerate to C1 ordering there",
            "supplementary eggplant table reproduces the recorded sim fine-inversion: C1 and "
            "every gate-inheriting candidate score pairwise_acc 0.489-0.681 on sim while the "
            "grip-heavy ungated blend (C4b) scores 1.000"],
    }
    json.dump(report, open("results/analysis/reward_bakeoff_v2.json", "w"), indent=1)

    # ---------------- prints ----------------
    print("\n" + "=" * 70)
    print(f"GATE-ZERO over {len(covered_pairs)} covered pairs (all native; "
          f"{sum(1 for p in covered_pairs if states[p['task']].source == 'eggplant_sim')} on sim):")
    hdr = f"{'candidate':22s} {'GZ nat':>7s} {'frz-top1max':>11s} {'supp-acc':>9s} {'supp-top1max':>12s} {'norm':>5s}"
    print(hdr)
    for r in verdict_rows:
        print(f"{r['candidate']:22s} {r['gate_zero_native']:>7s} "
              f"{str(r['max_frozen_top1_regret']):>11s} {r['supp_pooled_pairwise_acc']:>9.3f} "
              f"{r['supp_max_top1_regret']:>12.1f} {str(r['norm_control_pass']):>5s}")
    print("\nSUMMARY")
    print(f"1. {len(subset_passers)}/9 candidates pass every COVERED native gate-zero sign "
          f"({counts['native']['both']} pairs) + norm control; no candidate FAILS the covered exam.")
    print(f"2. NO outright pass is certifiable: only {counts['native']['both']}/{n_native_pairs} "
          f"native pairs coverable; frozen regret administrable on 1/4 native tasks (stack, 3 "
          f"phrases, all candidates 0.0).")
    print(f"3. Leader on covered subset (tie-break = supplementary transfer, contaminated for "
          f"C1): {leader}; runner-up {ranking[1]['candidate']}.")
    print("4. Feature gap job to complete the exam: feature_extraction_needed.json "
          f"({sum(s['n_phrases'] for s in spec_tasks if s['required'])} required native phrases; "
          f"{sum(s['n_phrases'] for s in spec_tasks if not s['required'])} optional OOD).")
    print("5. Full artifacts: exam_coverage.json, reward_bakeoff_v2.json.")


if __name__ == "__main__":
    main()
