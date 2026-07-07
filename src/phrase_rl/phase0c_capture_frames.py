"""Capture each SIMPLER Bridge task's initial frame -> contexts-like parquet.

Output feeds gemini_rephrase.py (columns: episode_index, t, instruction,
image_png) so the 0c phrase lists are generated from the actual sim boot frames.

Run from INT-ACT root with its venv:
  .venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_capture_frames.py \
    --out /workspace/phrase-rl/data/contexts_0c_tasks.parquet
"""

import argparse
import io

import numpy as np
import pandas as pd
from PIL import Image

TASKS = [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    rows = []
    for i, task in enumerate(TASKS):
        env = simpler_env.make(task)
        obs, _ = env.reset(seed=args.seed, options={"obj_init_options": {"episode_id": 0}})
        instruction = env.unwrapped.get_language_instruction()
        img = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        rows.append(
            {"episode_index": i, "t": 0, "task": task, "instruction": instruction,
             "image_png": buf.getvalue(), "max_steps": env.spec.max_episode_steps}
        )
        print(f"{task}: {instruction!r} (max_steps={env.spec.max_episode_steps})")
        env.close()

    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} task contexts to {args.out}")


if __name__ == "__main__":
    main()
