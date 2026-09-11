"""Phase 0c rollout runner: (task, phrase, episode_id) -> success in SIMPLER.

Mirrors INT-ACT's SimplerEvaluator loop exactly (receding horizon, chunk of 4,
1 action per env step) but in-process (no websocket) and with OUR phrase in
element["task"]. Shared initial states across phrases = same episode_id list
(seed fixed; episode_id controls object placement — INT-ACT convention).

Records success + partial-progress stats (grasp, moved-correct-obj) per
episode; partial stats matter because binary success at n=10 seeds is coarse.

Run from /workspace/INT-ACT with its .venv (NOT `uv run` — it re-syncs):
  .venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
    --int-act-root /workspace/INT-ACT \
    --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
    --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
    --phrases /workspace/phrase-rl/data/phrases_0c.parquet \
    --episode-ids 0 1 2 3 4 5 6 7 8 9 \
    --out /workspace/phrase-rl/data/rollouts_0c.parquet
"""

import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int-act-root", required=True)
    ap.add_argument("--config", required=True, help="relative to int-act-root")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--phrases", required=True, help="parquet: task, arm, phrase")
    ap.add_argument("--episode-ids", type=int, nargs="+", required=True,
                    help="INT-ACT mode: obj_init episode ids. CoVer mode: trial indices (initial state = reset seed 1000 + i%%50)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cover-protocol", action="store_true",
                    help="mirror CoVer's Table-3 eval (audited from their code 2026-07-09): "
                    "initial states via env.reset(seed=1000+trial%%50) with NO obj_init_options, "
                    "run to --max-steps ignoring TimeLimit truncation, break on success")
    ap.add_argument("--max-steps", type=int, default=150, help="cover-protocol horizon (their loop: 150)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="rollouts per (task, phrase, episode_id) — pi0 decoding is stochastic, "
                    "so repeats measure policy noise (identical-phrase cells differed 8-16pp at n=25)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--record-dir", default=None, help="save per-episode trajectories (frames/states/executed actions) as npz — sim-grounded a* source")
    ap.add_argument("--record-success-only", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.int_act_root)
    os.chdir(root)  # their configs use relative paths (dataset statistics etc.)
    sys.path.insert(0, root)

    import draccus
    import simpler_env
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    from src.agent.configuration_pipeline import TrainPipelineConfig
    from src.experiments.policies.policy_wrapper import LeRobotPolicyWrapper

    pipeline_cfg = draccus.parse(
        TrainPipelineConfig,
        args.config,
        args=["--eval_cfg.pretrained_model_path", args.ckpt, "--seed", str(args.seed)],
    )
    policy = LeRobotPolicyWrapper(pipeline_cfg=pipeline_cfg, model_class=PI0Policy)
    policy._initialze_model_server(model_path=args.ckpt)  # sic — INT-ACT's method name
    policy.env_adapter = policy._initialize_env_adapter()  # only switch_model does this normally
    action_step = policy.action_step

    phrases = pd.read_parquet(args.phrases)
    done_rows = []
    done_keys = set()
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        done_rows = prev.to_dict("records")
        # older files have no rep column: treat their episodes as rep 0
        done_keys = {(r["task"], r["phrase"], r["episode_id"], r.get("rep", 0)) for r in done_rows}
        print(f"resume: {len(done_rows)} episodes already recorded")

    n_new = 0
    for task, task_phrases in phrases.groupby("task"):
        env = simpler_env.make(task)
        for row in task_phrases.itertuples():
            # optional per-row episode pinning (e.g. random-rephrase arm: one sampled
            # phrase per episode, CoVer's "pi0 w/ random" analog)
            row_eps = args.episode_ids
            if hasattr(row, "episode_id") and row.episode_id is not None and not pd.isna(row.episode_id):
                row_eps = [int(row.episode_id)]
            for ep_id in row_eps:
              for rep in range(args.repeats):
                if (task, row.phrase, ep_id, rep) in done_keys:
                    continue
                if args.cover_protocol:
                    # CoVer: itertools.count(1000) reset every 50 trials -> seeds 1000..1049 cycled,
                    # placement from the env's episode RNG (no obj_init_options)
                    obs, reset_info = env.reset(seed=1000 + ep_id % 50)
                else:
                    obs, reset_info = env.reset(
                        seed=args.seed, options={"obj_init_options": {"episode_id": ep_id}}
                    )
                policy.reset()  # resets model action queue + adapter
                # CRN for evals (2026-07-11): pin pi0's decode noise per (task, state, rep)
                # — phrase deliberately EXCLUDED, so all arms face identical noise and
                # arm contrasts reflect phrasing only (identical-phrase cells differed
                # 8-16pp without this)
                import zlib
                import torch as _torch
                _seed = zlib.crc32(f"{task}|{ep_id}|{rep}".encode()) % (2**31)
                _torch.manual_seed(_seed)
                if _torch.cuda.is_available():
                    _torch.cuda.manual_seed_all(_seed)
                action_plan = collections.deque()
                success, steps = False, 0
                ever_success = False  # first-frame metric, recorded alongside final-state
                first_success_step = None
                rec = {"imgs": [], "states": [], "actions": []} if args.record_dir else None
                while True:
                    img = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))
                    if rec is not None:
                        import io as _io
                        from PIL import Image as _Image
                        from transforms3d.euler import quat2euler as _q2e
                        _b = _io.BytesIO(); _Image.fromarray(img).save(_b, format="PNG")
                        rec["imgs"].append(_b.getvalue())
                        _e = np.asarray(obs["agent"]["eef_pos"], dtype=np.float32)  # xyz, quat(wxyz), gripper
                        _rpy = _q2e(_e[3:7], axes="sxyz")
                        rec["states"].append(np.array([*_e[:3], *_rpy, 0.0, _e[7]], dtype=np.float32))  # Bridge convention
                    if not action_plan:
                        element = {
                            "observation.images.top": img,
                            "observation.state": obs,
                            "task": str(row.phrase),
                        }
                        action_chunk = policy.select_action(element)
                        action_plan.extend(action_chunk[:action_step])
                    action = action_plan.popleft()
                    if rec is not None:
                        rec["actions"].append(np.asarray(action, dtype=np.float32))
                    obs, reward, success, truncated, info = env.step(action.copy())
                    if bool(success) and first_success_step is None:
                        first_success_step = steps
                    ever_success = ever_success or bool(success)
                    steps += 1
                    if args.cover_protocol:
                        # their loop: break on success, ignore TimeLimit, hard cap at max_steps
                        if success or steps >= args.max_steps:
                            break
                    elif truncated:
                        break
                if rec is not None and (success or not args.record_success_only):
                    import os as _os
                    _os.makedirs(args.record_dir, exist_ok=True)
                    _safe = "".join(c if c.isalnum() else "_" for c in str(row.phrase))[:40]
                    np.savez_compressed(
                        f"{args.record_dir}/{task}__ep{ep_id}__{_safe}.npz",
                        imgs=np.array(rec["imgs"], dtype=object), states=np.stack(rec["states"]),
                        actions=np.stack(rec["actions"]), success=success,
                        task=task, phrase=str(row.phrase), episode_id=ep_id)
                stats = info.get("episode_stats", {})
                done_rows.append(
                    {
                        "task": task, "arm": row.arm, "phrase": row.phrase,
                        "episode_id": ep_id, "rep": rep, "success": bool(success), "steps": steps,
                        "ever_success": bool(ever_success),
                        "first_success_step": first_success_step,
                        "grasped": int(stats.get("is_src_obj_grasped", 0)),
                        "moved_correct": int(stats.get("moved_correct_obj", 0)),
                        "moved_wrong": int(stats.get("moved_wrong_obj", 0)),
                    }
                )
                n_new += 1
                if n_new % args.save_every == 0:
                    pd.DataFrame(done_rows).to_parquet(args.out, index=False)
                    print(f"[{n_new}] {task} ep{ep_id} success={success} phrase={row.phrase[:40]!r}")
        env.close()

    pd.DataFrame(done_rows).to_parquet(args.out, index=False)
    df = pd.DataFrame(done_rows)
    print(f"done: {n_new} new episodes ({len(df)} total) -> {args.out}")
    print(df.groupby("task")["success"].mean())


if __name__ == "__main__":
    main()
