#!/usr/bin/env python3
"""Build the 8-task probe files (4 existing ID tasks + 4 OOV val scenes).

  --stage render    (pod, INT-ACT venv): first frames of the 4 OOV val envs
  --stage traces    (local, genai key): CoVer teacher-trace call per new scene
  --stage assemble  (local): contexts_probe8.parquet + traces_probe8.parquet
"""
import argparse
import io
import sys

import pandas as pd

VAL4 = {"widowx_carrot_on_keyboard_clean": "put carrot on keyboard",
        "widowx_carrot_on_wheel_clean": "put carrot on wheel",
        "widowx_coke_can_on_ramekin_clean": "put coke can on ramekin",
        "widowx_coke_can_on_plate_clean": "put coke can on plate"}
EP0 = {t: 990001 + i for i, t in enumerate(sorted(VAL4))}
FRAMES = "data/probe8_frames.parquet"


def stage_render():
    import numpy as np
    from PIL import Image
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    rows = []
    for task, nominal in VAL4.items():
        env = simpler_env.make(task)
        obs, _ = env.reset(seed=0, options={"obj_init_options": {"episode_id": 0}})
        img = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        rows.append({"task": task, "nominal": nominal, "image_png": buf.getvalue()})
        env.close()
        print("rendered", task)
    pd.DataFrame(rows).to_parquet(FRAMES, index=False)


def stage_traces():
    sys.path.insert(0, "src")
    from google import genai
    from google.genai import types
    from phrase_rl.cover_prompt import USER_TEMPLATE
    from phrase_rl.gemini_redteam_assets import call_with_retry
    client = genai.Client()
    fr = pd.read_parquet(FRAMES)
    rows = []
    for r in fr.itertuples():
        prompt = USER_TEMPLATE.format(instruction=r.nominal, batch_number=16)
        img_part = types.Part.from_bytes(data=bytes(r.image_png), mime_type="image/png")
        text = call_with_retry(client, types, "gemini-3.5-flash", [img_part, prompt],
                               temperature=0.8, max_tokens=2000)
        trace = text.split("Reworded Instructions")[0].rstrip().rstrip(":").rstrip()
        # normalize markdown-header drift to the canonical teacher-trace tags
        import re
        trace = re.sub(r"\*\*Description of the image:?\*\*:?", "<Description of the image>", trace)
        trace = re.sub(r"\*\*Meaning of the instruction in the context of the image:?\*\*:?",
                       "<Meaning of the instruction in the context of the image>", trace)
        trace = trace.replace("**", "")
        assert "<Description of the image>" in trace, f"unexpected trace format for {r.task}"
        rows.append({"episode_index": EP0[r.task], "t": 0, "trace": trace})
        print(f"trace for {r.task}: {len(trace)} chars")
    pd.DataFrame(rows).to_parquet("data/probe8_traces_new.parquet", index=False)


def stage_assemble():
    ctx = pd.read_parquet("results/phrase_artifacts/contexts_0c_tasks.parquet")
    tra = pd.read_parquet("results/phrase_artifacts/traces_0c_tasks.parquet")
    fr = pd.read_parquet(FRAMES)
    newc = pd.DataFrame([{"episode_index": EP0[r.task], "t": 0, "task": r.task,
                          "instruction": r.nominal, "image_png": r.image_png,
                          "max_steps": int(ctx.max_steps.max())} for r in fr.itertuples()])
    newt = pd.read_parquet("data/probe8_traces_new.parquet")
    pd.concat([ctx, newc], ignore_index=True).to_parquet(
        "results/phrase_artifacts/contexts_probe8.parquet", index=False)
    pd.concat([tra, newt], ignore_index=True).to_parquet(
        "results/phrase_artifacts/traces_probe8.parquet", index=False)
    print(f"probe8: {len(ctx) + len(newc)} contexts, {len(tra) + len(newt)} traces")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["render", "traces", "assemble"])
    a = ap.parse_args()
    {"render": stage_render, "traces": stage_traces, "assemble": stage_assemble}[a.stage]()
