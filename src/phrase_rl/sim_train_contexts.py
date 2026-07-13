"""Render sim training contexts for the rollout-reward arm: one row per
(task, episode_id) = the initial frame + nominal instruction of each layout.

96 contexts (4 tasks x 24 layouts). The rollout reward judges each candidate
phrase on the SAME layout it conditions on (context_id = "task|episode_id").

Run on a SIMPLER pod (INT-ACT venv):
  /workspace/INT-ACT/.venv/bin/python -m phrase_rl.sim_train_contexts \
    --int-act-root /workspace/INT-ACT --out data/contexts_sim_rollout.parquet
"""

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

TASKS = [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int-act-root", required=True)
    ap.add_argument("--episode-ids", type=int, nargs="+", default=list(range(24)))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = os.path.abspath(args.int_act_root)
    os.chdir(root)
    sys.path.insert(0, root)
    import simpler_env
    from PIL import Image
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    rows = []
    for task in TASKS:
        env = simpler_env.make(task)
        instruction = None
        for ep in args.episode_ids:
            obs, _ = env.reset(seed=args.seed, options={"obj_init_options": {"episode_id": ep}})
            if instruction is None:
                instruction = env.unwrapped.get_language_instruction()
            img = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            # per-task offset: (episode_index, t) must be unique ACROSS tasks or the
            # trainer's trace dict collapses (all tasks share ep numbers 0-23)
            off = 1000 * (1 + TASKS.index(task))
            rows.append({
                "context_id": f"{task}|{ep}",
                "task": task, "episode_index": off + ep, "t": 0,
                "episode_id": ep, "instruction": str(instruction),
                "image_png": buf.getvalue(),
            })
        env.close()
        print(f"{task}: {len(args.episode_ids)} layouts, instruction={instruction!r}")
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} sim contexts -> {args.out}")


if __name__ == "__main__":
    main()
