"""OOV-hope probe: Qwen self-traces (CoVer prompt, image + hostile ERT) for the
wheel/tire and ramekin tasks, printed beside the Gemini traces from the assets.
Question: does 9B reasoning resolve the OOV receptacle to a usable noun?"""
import io
import json

import pandas as pd
import torch
from PIL import Image

import sys
sys.path.insert(0, "src")
from phrase_rl.cover_prompt import build_qwen_messages, extract_trace
from phrase_rl.phase2_train import apply_template

TASKS = ["widowx_carrot_on_wheel_clean", "widowx_coke_can_on_ramekin_clean"]

assets = pd.read_parquet("data/val_screen_assets.parquet")
frames = pd.read_parquet("data/val_screen_frames.parquet")

from transformers import AutoModelForImageTextToText, AutoProcessor
proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B", dtype=torch.bfloat16,
                                                    device_map="cuda")
model.eval()

out = {}
for task in TASKS:
    a = assets[assets.task == task].iloc[0]
    fr = frames[frames.task == task].iloc[0]
    img = Image.open(io.BytesIO(fr["image_png"]))
    msgs = build_qwen_messages(img, str(a.ert_instruction), 1)
    enc = apply_template(proc, msgs).to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, do_sample=False, max_new_tokens=700)
    raw = proc.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    qtrace = extract_trace(raw)
    out[task] = {"ert": str(a.ert_instruction), "qwen_trace": qtrace,
                 "gemini_trace": str(a.trace)}
    print(f"\n{'='*90}\nTASK {task}\nERT: {a.ert_instruction}\n")
    print(f"--- QWEN 9B self-trace ---\n{qtrace}\n")
    print(f"--- GEMINI trace (assets) ---\n{a.trace}\n", flush=True)

json.dump(out, open("data/trace_probe_oov.json", "w"), indent=1)
print("TRACE-PROBE-DONE")
