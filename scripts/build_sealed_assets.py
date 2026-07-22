"""Sealed-test asset builder (fires only post-freeze; FINAL_EVAL=1 required
for sealed tasks; --dry-run exercises the pipeline on ONE val task instead).

Stage A (INT-ACT venv):   render initial frame (episode_id 0) per task
Stage B (.venv-gen):      Gemini batch — author ERT (image+nominal+genre shots)
                          + CoVer-template trace (image+ERT), per task
Stage C (.venv-gen):      Qwen self-trace per task (sft17k_selftrace_gen)

Stages are selected by --stage {render,gemini,selftrace}; the wrapper script
runs them in order under the right venvs. Outputs:
  data/sealed_frames.parquet          (task, nominal, image_png)
  data/sealed_assets_gemini.parquet   (task, ert_instruction, trace)
  data/sealed_assets_selftrace.parquet(task, ert_instruction, trace)
"""
import argparse
import io
import json
import os
import sys

SEALED = {
 "widowx_carrot_on_ramekin_clean": "put carrot on ramekin",
 "widowx_carrot_on_sponge_clean": "put carrot on sponge",
 "widowx_coke_can_on_keyboard_clean": "put coke can on keyboard",
 "widowx_coke_can_on_wheel_clean": "put coke can on wheel",
 "widowx_eggplant_on_keyboard_clean": "put eggplant on keyboard",
 "widowx_eggplant_on_sponge_clean": "put eggplant on sponge",
 "widowx_green_cube_on_plate_clean": "put green cube on plate",
 "widowx_nut_on_plate_clean": "put nut on plate",
 "widowx_nut_on_wheel_clean": "put nut on wheel",
 "widowx_orange_juice_on_plate_clean": "put orange juice on plate",
 "widowx_pepsi_can_on_plate_clean": "put pepsi can on plate",
 "widowx_small_plate_on_green_cube_clean": "put small plate on green cube",
}
DRY = {"widowx_spoon_on_towel": "put the spoon on the towel"}


def tasks(args):
    if args.dry_run:
        return DRY
    if os.environ.get("FINAL_EVAL") != "1":
        raise SystemExit("sealed tasks require FINAL_EVAL=1")
    return SEALED


def stage_render(args):
    import numpy as np
    import pandas as pd
    from PIL import Image
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    rows = []
    for task, nominal in tasks(args).items():
        env = simpler_env.make(task)
        obs, _ = env.reset(seed=0, options={"obj_init_options": {"episode_id": 0}})
        img = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        rows.append({"task": task, "nominal": nominal, "image_png": buf.getvalue()})
        env.close()
        print(f"[render] {task} ok", flush=True)
    pd.DataFrame(rows).to_parquet(args.frames_out, index=False)
    print(f"wrote {len(rows)} -> {args.frames_out}", flush=True)


ERT_PROMPT = (
    "You are building an Extreme Rephrasing Test (ERT) for robot instructions: "
    "adversarial rewordings that keep the task goal but are maximally hostile to a "
    "language-conditioned robot policy. The attached image shows the actual scene. "
    "Reference objects by description or indirect means rather than their plain "
    "names, add scene-plausible attributes, keep the SAME goal. Examples of the genre:\n"
    "{shots}\n\nNominal instruction: \"{nominal}\"\n"
    "Produce exactly ONE complete ERT rephrasing (output just the instruction, "
    "no quotes, no explanation)."
)


def stage_gemini(args):
    import pandas as pd
    sys.path.insert(0, "src")
    from phrase_rl.gemini_redteam_assets import call_with_retry
    from phrase_rl.cover_prompt import build_user_prompt, load_system_prompt  # noqa: F401
    from google import genai
    from google.genai import types
    from phrase_rl.cover_prompt import extract_trace
    cover = json.load(open("results/phrase_artifacts/cover_release_ert_rephrases.json"))["instructions"]
    shots = "\n".join(f'nominal: "{n}" -> ERT: "{r["ert_rephrases"][0]}"'
                      for n, r in list(cover.items())[:4])
    client = genai.Client()
    frames = pd.read_parquet(args.frames_out)
    rows = []
    for r in frames.itertuples():
        img_part = types.Part.from_bytes(data=r.image_png, mime_type="image/png")
        ert = call_with_retry(client, types, "gemini-3.5-flash",
                              [img_part, ERT_PROMPT.format(shots=shots, nominal=r.nominal)],
                              temperature=0.8, max_tokens=400).strip().strip('"')
        ert = ert.split("\n")[0].strip()
        # trace: verbatim CoVer template on (image, ERT) — same shape as val assets
        trace_raw = call_with_retry(client, types, "gemini-3.5-flash",
                                    [img_part, build_user_prompt(ert, 1)],
                                    system=load_system_prompt(), temperature=0.4,
                                    max_tokens=2000)
        trace = extract_trace(trace_raw) or trace_raw[:1200]
        rows.append({"task": r.task, "nominal": r.nominal,
                     "ert_instruction": ert, "trace": trace})
        print(f"[gemini] {r.task}\n  ERT: {ert!r}\n  trace head: {trace[:120]!r}", flush=True)
    pd.DataFrame(rows).to_parquet(args.gemini_out, index=False)
    print(f"wrote {len(rows)} -> {args.gemini_out}", flush=True)


def stage_selftrace(args):
    # reuse the minimal-prompt generator against the sealed ERTs + frames
    import pandas as pd
    sys.path.insert(0, "src")
    ga = pd.read_parquet(args.gemini_out)[["task", "ert_instruction"]]
    fr = pd.read_parquet(args.frames_out)[["task", "image_png"]]
    ga.to_parquet("data/_sealed_st_assets.parquet", index=False)
    fr.to_parquet("data/_sealed_st_frames.parquet", index=False)
    from phrase_rl import sft17k_selftrace_gen as gen
    sys.argv = ["x", "--assets", "data/_sealed_st_assets.parquet",
                "--frames", "data/_sealed_st_frames.parquet",
                "--out", args.selftrace_out]
    gen.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["render", "gemini", "selftrace"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--frames-out", default="data/sealed_frames.parquet")
    ap.add_argument("--gemini-out", default="data/sealed_assets_gemini.parquet")
    ap.add_argument("--selftrace-out", default="data/sealed_assets_selftrace.parquet")
    args = ap.parse_args()
    if args.dry_run:
        args.frames_out = args.frames_out.replace("sealed", "dryrun")
        args.gemini_out = args.gemini_out.replace("sealed", "dryrun")
        args.selftrace_out = args.selftrace_out.replace("sealed", "dryrun")
    {"render": stage_render, "gemini": stage_gemini, "selftrace": stage_selftrace}[args.stage](args)


if __name__ == "__main__":
    main()
