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
    ap.add_argument("--episode-ids", type=int, nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-every", type=int, default=20)
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
    action_step = policy.action_step

    phrases = pd.read_parquet(args.phrases)
    done_rows = []
    done_keys = set()
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        done_rows = prev.to_dict("records")
        done_keys = {(r["task"], r["phrase"], r["episode_id"]) for r in done_rows}
        print(f"resume: {len(done_rows)} episodes already recorded")

    n_new = 0
    for task, task_phrases in phrases.groupby("task"):
        env = simpler_env.make(task)
        for row in task_phrases.itertuples():
            for ep_id in args.episode_ids:
                if (task, row.phrase, ep_id) in done_keys:
                    continue
                obs, reset_info = env.reset(
                    seed=args.seed, options={"obj_init_options": {"episode_id": ep_id}}
                )
                policy.model.reset()
                action_plan = collections.deque()
                success, steps = False, 0
                while True:
                    img = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))
                    if not action_plan:
                        element = {
                            "observation.images.top": img,
                            "observation.state": obs,
                            "task": str(row.phrase),
                        }
                        action_chunk = policy.select_action(element)
                        action_plan.extend(action_chunk[:action_step])
                    action = action_plan.popleft()
                    obs, reward, success, truncated, info = env.step(action.copy())
                    steps += 1
                    if truncated:
                        break
                stats = info.get("episode_stats", {})
                done_rows.append(
                    {
                        "task": task, "arm": row.arm, "phrase": row.phrase,
                        "episode_id": ep_id, "success": bool(success), "steps": steps,
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
