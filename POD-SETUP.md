# POD-SETUP — provisioning a RunPod worker that can render (SIMPLER/Vulkan)

The distilled runbook from pods 1–4 of this project. Follow it top to bottom
for a new pod; every step exists because skipping it burned us once.

## 0. Provisioning requirements (before you rent it)

- **Same template/image family as the existing pods.** The non-negotiable
  property: the container must be granted the `graphics` driver capability.
- 48GB-class GPU for training+scoring on one card (A6000 / L40 / L40S; L40S
  ≈ 2× A6000 for bf16 training). 24GB cards cannot host trainer + score
  server together.
- Network-volume (/workspace) pods: check the **volume quota** vs payload
  size — extracted clone + tars can exceed 120G transiently (see §5).

## 1. First contact — verify graphics BEFORE building anything

```bash
tr "\0" "\n" < /proc/1/environ | grep NVIDIA_DRIVER_CAPABILITIES
```

`/proc/1/environ` is the AUTHORITY (an empty shell env var is meaningless).
It must include `graphics`. **If it doesn't, STOP and recreate the pod** —
this cannot be fixed from inside. Everything else below is fixable.

## 2. Base prep (apt silently fails without update — check exit codes!)

```bash
apt-get update -qq
apt-get install -y rsync tmux libvulkan1 libegl1 libgles2 vulkan-tools
git config --global user.email "<email>"   # unset identity = SILENT commit failures
git config --global user.name  "<name>"
```

Footguns encoded here:
- `apt-get install` before `update` on a fresh image fails, and `-qq ... >/dev/null; echo OK`
  hides it. Never chain-suppress the first apt call.
- `libegl1`/`libgles2` are REQUIRED even headless — the NVIDIA ICD returns
  NULL without the glvnd EGL dispatcher.

## 3. Vulkan — the three-tier diagnosis

| symptom | cause | fix |
|---|---|---|
| `vulkaninfo` shows only llvmpipe; sapien: `ErrorExtensionNotPresent` | loader only reads `/usr/share/vulkan/icd.d/` and the NVIDIA manifest is only in `/etc/vulkan/icd.d/` | `cp /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/` — keep it in BOTH paths |
| no NVIDIA manifest anywhere | image shipped without it | write it: `{"file_format_version":"1.0.1","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.4.303"}}` into both icd.d dirs |
| manifests fine, still no device | `graphics` missing from `/proc/1/environ` | unfixable — recreate the pod (§1) |

Verify: `vulkaninfo --summary` must list the real GPU (llvmpipe alongside is
fine). Then the true smoke test:

```bash
cd /workspace/INT-ACT && .venv/bin/python -c "
import simpler_env
env = simpler_env.make('widowx_spoon_on_towel')
obs,_ = env.reset(seed=0, options={'obj_init_options':{'episode_id':0}})
print('RENDER-SMOKE-OK'); env.close()"
```

(`GLFW error: X11 DISPLAY missing` in the output is normal headless noise.)
Note: pod restarts/resizes can DROP the manifest mirror — re-run §3 checks
after any restart.

## 4. The three venvs (and why each breaks)

| venv | lives at | used for | rebuild |
|---|---|---|---|
| INT-ACT rollout stack | `/workspace/INT-ACT/.venv` | phase0c rollouts, SIMPLER | ships inside INT-ACT dir; survives with /workspace |
| `.venv` → `/root/venv` | container disk | score server (lerobot PI0) | recipe in `scripts/run_verifier_data_multit.sh` §venv block — python 3.11, `-e .[gpu]`, lerobot IrvingF7 fork @35f6e02, transformers==4.48.3, **and `pytest av`** (the fork literally does `from pytest import Cache`) |
| `.venv-gen` → `/root/venv-gen` | container disk | Qwen gen/train | `scripts/rebuild_gen_venv.sh`, or rsync `/root/venv-gen` from a live pod **to the same absolute path** |

Footguns:
- **Container disk (`/root`) does not survive pod stop/restart.** After any
  restart, assume `/root/venv*` are gone until proven otherwise (pod1 lost
  `/root/venv` this way and nothing noticed until the score server needed it).
- Repo symlinks `.venv`/`.venv-gen` point at ABSOLUTE `/root/...` paths —
  restore venvs to the SAME path or the symlinks dangle.
- venv tars have RELATIVE members (`venv-gen/...`): extract with `-C /root`,
  NOT `-C /` (that lands at `/venv-gen` and the symlink dangles anyway).

## 5. Payloads — copy vs rebuild

Ship from a live pod (rsync over a **pod-scoped relay key**, never your
personal key: `ssh-keygen -f ~/.ssh/pod_relay` on the sender, append the
.pub to the receiver's authorized_keys):
- `/workspace/hf_cache` (~25G — π0 + Qwen weights)
- `/workspace/INT-ACT` (~11G incl. its venv)
- `/root/venv`, `/root/venv-gen` (same-path, see §4)
- `data/` payloads not in git (context tables etc. — check
  `gitignored-data-payload` discipline before ANY pod retirement)

Always REBUILD from git, never copy: the repo itself (`git clone`), plus
anything under `results/` (checkpoint archive restores via
`scripts/ckpt_archive.sh unpack <name>`).

Quota discipline: delete transfer tars IMMEDIATELY after extraction —
`Disk quota exceeded` on mfs volumes strikes mid-`git fetch` and even
`git config` writes fail (the 4GB checkpoint-archive fetch needs headroom).
Also `git config fetch.unpackLimit 1` helps big fetches on slow volumes.

## 6. Secrets

`HF_TOKEN`, `GEMINI_API_KEY` etc. travel as a FILE (scp a bashrc fragment,
`cat >> ~/.bashrc`), NEVER in echoed commands or logs. Scripts source them
via `eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc)"`.

## 7. Smoke-test checklist before real work

1. `vulkaninfo --summary` → real GPU listed
2. INT-ACT `env.reset` → RENDER-SMOKE-OK (§3)
3. `/root/venv/bin/python -c "from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy"` (falls back to `lerobot.policies...` on ≥0.4 layouts)
4. `.venv-gen/bin/python -c "import torch"`
5. `cd /workspace/phrase-rl && git pull && git commit --allow-empty -m probe && git push` (then reset) — proves identity + credentials + quota
6. First worker board: watch the FIRST result land before walking away —
   silent stalls (mfs D-state) and half-dead shard sets both happened.

## 8. Retirement checklist (BEFORE shutting a pod down)

1. `results/checkpoints/` + `/workspace/adapters` → everything named is in
   the git archive (`ckpt_archive.sh pack`) — ALWAYS-archive rule.
2. `data/` inventory → irreplaceable parquets shipped or archived.
3. Any un-pushed commits? (`git status`, `git log origin/main..HEAD`)
4. tmux sessions doing anything? (`tmux ls`)
5. Only then: stop the pod. (And remember: stopping wipes `/root`.)
