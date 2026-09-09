#!/usr/bin/env python3
"""A37 harness adaptations to CoVer's released eval (exact-string patches).

Applied on the CoVer pod at /workspace/cover-vla/CoVer_VLA. Idempotent; every
anchor must match exactly once or nothing is written. Mirror copy kept in
phrase-rl under reference/cover_patches/ for provenance.

Adds: (1) the simpler_sealed12 suite (our 12 INT-ACT envs); (2) flags —
rephrase_json (wave files in their schema), pin_layouts (phase0c-style resets:
seed=42 + obj_init_options episode_id, trial -> ep=trial%n, rep=trial//n),
pin_decode_noise (CRN torch seed crc32(task|ep|rep), Anchor B only),
results_jsonl (per-episode records), save_videos.
"""
import pathlib

SIM = pathlib.Path("/workspace/cover-vla/CoVer_VLA/inference/experiments/robot/simpler")


def patch(path, subs, tag):
    p = SIM / path
    src = p.read_text()
    if f"A37-{tag}" in src:
        print(f"already patched: {path}")
        return
    for old, new in subs:
        assert old in src, f"ANCHOR MISSING in {path}: {old[:80]!r}"
        assert src.count(old) == 1, f"ANCHOR NOT UNIQUE in {path}: {old[:80]!r}"
        src = src.replace(old, new)
    p.write_text(src)
    print(f"patched: {path}")


bench_add = '''

# ---- A37-suite (phrase-rl): the sealed 12 INT-ACT tasks ----
task_map["simpler_sealed12"] = [
    "widowx_carrot_on_ramekin_clean",
    "widowx_carrot_on_sponge_clean",
    "widowx_coke_can_on_keyboard_clean",
    "widowx_coke_can_on_wheel_clean",
    "widowx_cube_on_plate_clean",
    "widowx_eggplant_on_keyboard_clean",
    "widowx_eggplant_on_sponge_clean",
    "widowx_nut_on_plate_clean",
    "widowx_nut_on_wheel_clean",
    "widowx_orange_juice_on_plate_clean",
    "widowx_pepsi_on_plate_clean",
    "widowx_small_plate_on_green_cube_clean",
]


@register_benchmark
class SIMPLER_SEALED12(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_sealed12"
        self._make_benchmark()
'''
b = SIM / "simpler_benchmark.py"
src = b.read_text()
if "A37-suite" not in src:
    b.write_text(src + bench_add)
    print("patched: simpler_benchmark.py")
else:
    print("already patched: simpler_benchmark.py")

patch("run_simpler_eval_with_openpi.py", [
    ("import itertools",
     "import itertools\nimport json  # A37-eval\nimport zlib"),
    ("""    # Batch inference parameters
    policy_batch_inference_size: int = 2""",
     """    # Batch inference parameters
    policy_batch_inference_size: int = 2

    # A37 (phrase-rl) harness adaptations
    rephrase_json: Optional[str] = None      # wave JSON overriding the shipped rephrase file
    pin_layouts: bool = False                # reset like our phase0c: seed=42 + obj_init_options episode_id
    layouts_n: int = 24                      # trial -> (episode_id=trial%n, rep=trial//n)
    pin_decode_noise: bool = False           # CRN torch seed crc32(task|ep|rep) per episode (Anchor B)
    results_jsonl: Optional[str] = None      # per-episode records
    save_videos: bool = True"""),
    ("    preloaded_rephrases = load_rephrases(cfg.task_suite_name)",
     """    if cfg.rephrase_json:
        with open(cfg.rephrase_json) as _f:
            preloaded_rephrases = json.load(_f)["instructions"]
    else:
        preloaded_rephrases = load_rephrases(cfg.task_suite_name)"""),
    ("            obs, reset_info = env.reset(seed=next(seeds))",
     """            if cfg.pin_layouts:
                _ep_id = trial_idx % cfg.layouts_n
                _rep = trial_idx // cfg.layouts_n
                obs, reset_info = env.reset(
                    seed=42, options={"obj_init_options": {"episode_id": _ep_id}})
                if cfg.pin_decode_noise:
                    _s = zlib.crc32(f"{task}|{_ep_id}|{_rep}".encode()) % (2**31)
                    torch.manual_seed(_s)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(_s)
            else:
                obs, reset_info = env.reset(seed=next(seeds))"""),
    ("            episode_data['success'] = done\n            episode_data['episode_length'] = t",
     """            episode_data['success'] = done
            episode_data['episode_length'] = t
            if cfg.results_jsonl:
                _rec = {"suite": cfg.task_suite_name, "task": task,
                        "nominal": original_task_description,
                        "base": episode_data['used_task_description'],
                        "episode_id": (trial_idx % cfg.layouts_n) if cfg.pin_layouts else None,
                        "rep": (trial_idx // cfg.layouts_n) if cfg.pin_layouts else None,
                        "trial_idx": trial_idx, "success": bool(done),
                        "steps": int(t), "use_verifier": bool(cfg.use_verifier),
                        "lang_rephrase_num": int(cfg.lang_rephrase_num),
                        "checkpoint": str(cfg.pretrained_checkpoint)}
                with open(cfg.results_jsonl, "a") as _f:
                    _f.write(json.dumps(_rec) + "\\n")"""),
    ("            save_rollout_video_openpi(",
     "            (save_rollout_video_openpi if cfg.save_videos else (lambda *a, **k: None))("),
], "eval")
print("ALL PATCHES OK")
