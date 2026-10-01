#!/usr/bin/env python3
"""bank_eval.py with per-step frame capture: rolls each (suite, task_id,
phrase, init) in the queue and writes one npz per episode (imgs = PNG bytes
per step, success, steps) for the side-by-side grid video. Same env, seeds,
settle/replan/resize as bank_eval, so the outcomes replay the measured ones.

  MUJOCO_GL=egl python3 video_eval.py --queue queue.json --out /workspace/grid_frames
"""
import argparse, collections, io, json, os, pathlib, time
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy as _wcp
from fourtier_eval import _obs_element, ENV_RES, MAX_STEPS, DUMMY


def rollout_rec(env, init_state, prompt, client, max_steps, settle, replan, size):
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(settle):
        obs, _, _, _ = env.step(DUMMY)
    plan = collections.deque(); frames = []
    def snap(o):
        b = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(o["agentview_image"][::-1, ::-1])).save(b, "PNG")
        frames.append(b.getvalue())
    snap(obs)
    for step in range(max_steps):
        if not plan:
            chunk = client.infer(_obs_element(obs, prompt, size))["actions"]
            plan.extend(chunk[:replan])
        obs, _, done, _ = env.step(plan.popleft().tolist())
        snap(obs)
        if done:
            return True, step + 1, frames
    return False, max_steps, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", required=True); p.add_argument("--out", required=True)
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--settle", type=int, default=10); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--resize", type=int, default=224); p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    queue = json.load(open(a.queue))
    client = _wcp.WebsocketClientPolicy(a.host, a.port)
    suites, cur = {}, {"key": None, "env": None, "inits": None}
    def env_for(suite, tid):
        if cur["key"] == (suite, tid): return cur["env"], cur["inits"]
        if cur["env"] is not None: cur["env"].close()
        if suite not in suites: suites[suite] = benchmark.get_benchmark_dict()[suite]()
        task = suites[suite].get_task(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RES, camera_widths=ENV_RES)
        env.seed(a.seed)
        cur.update(key=(suite, tid), env=env, inits=suites[suite].get_task_init_states(tid))
        return cur["env"], cur["inits"]
    n = 0; total = sum(len(it["inits"]) for it in queue)
    for it in queue:
        suite, tid = it["suite"], it["task_id"]
        max_steps = max(MAX_STEPS.get(suite, 300), 300)
        safe = "".join(c if c.isalnum() else "_" for c in it["phrase"])[:40]
        for ii in it["inits"]:
            dst = out / f"{suite}__{tid}__{it['arm']}__{safe}__init{ii}.npz"
            if dst.exists(): n += 1; continue
            env, inits = env_for(suite, tid)
            t0 = time.time()
            ok, steps, frames = rollout_rec(env, inits[ii], it["phrase"], client, max_steps, a.settle, a.replan, a.resize)
            np.savez_compressed(dst, imgs=np.array(frames, dtype=object), success=ok, steps=steps,
                                task=f"{suite}/{tid}", phrase=it["phrase"], arm=it["arm"], init=ii)
            n += 1
            print(f"[{n}/{total}] {suite}/{tid} {it['arm']} init{ii} success={int(ok)} steps={steps} {time.time()-t0:.0f}s", flush=True)
    if cur["env"] is not None: cur["env"].close()
    print("VIDEO-QUEUE-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
