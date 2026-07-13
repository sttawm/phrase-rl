"""Rollout-reward server: SIMPLER success rates over the score-server file protocol.

The rollout-reward arm (user standing order 2026-07-10): reward = actual task
success in SIMPLER instead of an offline proxy. This process is the sim side:
it loads the frozen pi0 + the 4 ID envs once, then serves jobs from --ipc-dir.

Protocol (same shapes as phase2_score_server, one extra req field):
  req  : {job_id}.req.json {"in_parquet", "out_parquet", "reps": R}
  in   : one row per (context, phrase): context_id ("task|episode_id"), phrase
  out  : context_id, phrase, loss (= -success_rate  — LOWER IS BETTER to match
         the trainer's reward convention R = -loss), success_rate, n_eps
  done : req renamed to .done.json (atomic); failures -> .failed.json + .err.txt

Reward semantics: success_rate over R CRN-seeded reps of the row's OWN initial
state (context = a specific (task, episode_id) layout; the phrase is judged on
the layout it conditions on). Decode noise pinned per (task, ep, rep) with the
phrase EXCLUDED — advantages within a context compare phrases under identical
noise, the rollout analog of the CRN flow scoring.

Run on a SIMPLER-capable pod (INT-ACT venv), env vars per INT-ACT:
  /workspace/INT-ACT/.venv/bin/python -m phrase_rl.phase2_rollout_server \
      --int-act-root /workspace/INT-ACT --ipc-dir /workspace/ipc_rollout
"""

import argparse
import collections
import glob
import json
import os
import sys
import time
import traceback
import zlib

import numpy as np
import pandas as pd

DEFAULT_CKPT = "juexzz/INTACT-pi0-finetune-rephrase-bridge"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class RolloutWorker:
    """Owns the policy + a cached env per task; runs one episode at a time."""

    def __init__(self, int_act_root: str, ckpt: str, seed: int = 42):
        root = os.path.abspath(int_act_root)
        os.chdir(root)
        sys.path.insert(0, root)
        import draccus
        import simpler_env  # noqa: F401
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
        from src.agent.configuration_pipeline import TrainPipelineConfig
        from src.experiments.policies.policy_wrapper import LeRobotPolicyWrapper

        self._simpler_env = __import__("simpler_env")
        cfg = draccus.parse(
            TrainPipelineConfig,
            "config/experiment/simpler/pi0_finetune_bridge_ev.yaml",
            args=["--eval_cfg.pretrained_model_path", ckpt, "--seed", str(seed)],
        )
        self.policy = LeRobotPolicyWrapper(pipeline_cfg=cfg, model_class=PI0Policy)
        self.policy._initialze_model_server(model_path=ckpt)  # sic
        self.policy.env_adapter = self.policy._initialize_env_adapter()
        self.action_step = self.policy.action_step
        self.seed = seed
        self._envs = {}

    def env_for(self, task: str):
        if task not in self._envs:
            self._envs[task] = self._simpler_env.make(task)
        return self._envs[task]

    def episode(self, task: str, episode_id: int, phrase: str, rep: int) -> bool:
        import torch
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

        env = self.env_for(task)
        obs, _ = env.reset(seed=self.seed, options={"obj_init_options": {"episode_id": episode_id}})
        self.policy.reset()
        # CRN: pin decode noise per (task, state, rep) — phrase excluded
        s = zlib.crc32(f"{task}|{episode_id}|{rep}".encode()) % (2**31)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        plan = collections.deque()
        success = False
        while True:
            img = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))
            if not plan:
                chunk = self.policy.select_action(
                    {"observation.images.top": img, "observation.state": obs, "task": str(phrase)})
                plan.extend(chunk[: self.action_step])
            obs, _, success, truncated, _ = env.step(plan.popleft().copy())
            if truncated:
                return bool(success)


def process_job(worker: RolloutWorker, spec: dict, job_id: str):
    df = pd.read_parquet(spec["in_parquet"])
    reps = int(spec.get("reps", 2))
    out_rows = []
    for r in df.itertuples():
        task, ep = str(r.context_id).split("|")
        succ = [worker.episode(task, int(ep), str(r.phrase), k) for k in range(reps)]
        rate = float(np.mean(succ))
        out_rows.append({"context_id": r.context_id, "phrase": str(r.phrase),
                         "loss": -rate, "success_rate": rate, "n_eps": reps,
                         "loss_per_draw": [-float(s) for s in succ]})  # per-rep, trainer-reader compatible
    out = pd.DataFrame(out_rows)
    tmp = spec["out_parquet"] + ".tmp"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, spec["out_parquet"])
    return len(df), reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int-act-root", required=True)
    ap.add_argument("--ipc-dir", required=True)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--poll", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.ipc_dir, exist_ok=True)
    log("loading policy + envs (first episode also JIT-warms the sim)...")
    worker = RolloutWorker(args.int_act_root, args.ckpt)
    log(f"serving {args.ipc_dir}")
    while True:
        for req in sorted(glob.glob(os.path.join(args.ipc_dir, "*.req.json"))):
            base = req[: -len(".req.json")]
            job_id = os.path.basename(base)
            try:
                spec = json.load(open(req))
                t0 = time.time()
                n, reps = process_job(worker, spec, job_id)
                os.replace(req, base + ".done.json")
                log(f"job {job_id}: {n} phrases x {reps} reps in {time.time()-t0:.0f}s")
            except KeyboardInterrupt:
                raise
            except Exception:
                tb = traceback.format_exc()
                log(f"job {job_id} FAILED:\n{tb}")
                try:
                    open(base + ".err.txt", "w").write(tb)
                    os.replace(req, base + ".failed.json")
                except OSError:
                    pass
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
