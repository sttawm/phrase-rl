---
name: runpod-vulkan-rendering
description: How to check RunPod pods for Vulkan/graphics support before building SIMPLER or other render stacks
metadata: 
  node_type: memory
  type: reference
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
---

User has had repeated Vulkan trouble on RunPod (SIMPLER/SAPIEN rendering). Root cause is usually the container template, not the code: pods launched with `NVIDIA_DRIVER_CAPABILITIES=compute,utility` never mount the driver's graphics libs, and no in-pod apt install can fix it — the pod must be recreated with a template/env including `graphics` (or `all`).

**Check BEFORE building any render stack** (30 seconds):
```
tr "\0" "\n" < /proc/1/environ | grep NVIDIA_DRIVER_CAPABILITIES   # want graphics in the list
ls /usr/lib/x86_64-linux-gnu/ | grep -E "libGLX_nvidia|libEGL_nvidia"  # must exist
find /etc/vulkan /usr/share/vulkan -name "nvidia_icd*.json" 2>/dev/null
```
If graphics libs are missing → tell user to recreate the pod with env `NVIDIA_DRIVER_CAPABILITIES=all`; network-volume data survives.

The A40 pod in CA-MTL-1 (2026-07, phrase-rl Phase 0c) had full capabilities out of the box: `compute,display,graphics,utility,video` + ICD at /etc/vulkan/icd.d/. `NVIDIA_VISIBLE_DEVICES=void` at PID1 is normal on RunPod and does not mean no GPU.

Related: [[phrase-rl-experiment]] (if written later).

**2026-07-07 resolution (phrase-rl 0c):** SIMPLER/SAPIEN rendered fine on the CA-MTL-1 A40 once the *Python* stack was fixed — Vulkan itself was never the problem on this template. Real blockers: ruckig's stale scikit-build-core metadata (pin `scikit-build-core<0.10` + nanobind, `--no-build-isolation`), `setuptools<70` for pkg_resources, and the killer gotcha: **`uv run` re-syncs to the lockfile and silently uninstalls manually-added packages** — use `.venv/bin/python` directly or `uv run --no-sync`. Full recipe: `scripts/setup_simpler.sh` in the repo.
