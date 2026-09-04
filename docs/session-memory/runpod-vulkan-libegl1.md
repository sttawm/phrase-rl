---
name: runpod-vulkan-libegl1
description: NVIDIA Vulkan ICD silently fails without libegl1 — install it in every render-pod apt line
metadata: 
  node_type: memory
  type: project
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
---

On RunPod containers (drivers 550/570 era, 2026-07), the NVIDIA Vulkan ICD fails with
`loader_scanned_icd_add: Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr'
for ICD libGLX_nvidia.so.0` (only llvmpipe enumerates → SAPIEN dies with
`ErrorExtensionNotPresent`) whenever the glvnd EGL dispatcher **libEGL.so.1 is absent**.
The driver's init strace ends hunting libEGL.so.1 and never touches /dev. `libgl1` does
NOT pull it.

**Why:** the ICD initializes glvnd EGL internally before exposing Vulkan entry points.
Capability env (`NVIDIA_DRIVER_CAPABILITIES=graphics`), ICD manifests, device nodes,
restarts vs fresh creates — all irrelevant; diagnosed 2026-07-17 after chasing every
one of those (see [[runpod-vulkan-rendering]], [[runpod-vulkan-icd-manifest]]).

**How to apply:** every render-pod apt line must include `libegl1 libgles2`
(alongside libvulkan1 libgl1 libglib2.0-0 ffmpeg). Verify with
`vulkaninfo --summary | grep deviceName` → the GPU must enumerate, not just llvmpipe.
