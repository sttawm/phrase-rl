---
name: runpod-vulkan-icd-manifest
description: Two-tier vulkan repair on RunPod — ICD manifest fix vs unfixable missing graphics capability
metadata:
  type: reference
---

When SimplerEnv/sapien fails with `vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent` on a RunPod container:
1. If `/usr/share/vulkan/icd.d/nvidia_icd.json` is missing but `libGLX_nvidia.so.0` exists → write the manifest (`{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}`) — fixes the loader-can't-find-NVIDIA case.
2. If the error persists after the manifest, the container runtime lacks the graphics capability (env `NVIDIA_DRIVER_CAPABILITIES` empty/without `graphics`) — NOT fixable from inside the pod. Requires console-side fix: add env `NVIDIA_DRIVER_CAPABILITIES=all` to the pod config and restart, or recreate the pod. Volume resize restarts CAN silently drop this capability even when the pre-resize container had it (observed 2026-07-17 on pod 3/A5000).

Related: [[runpod-vulkan-rendering]] (original check), [[runpod-podstate-backup]] (restore.sh should also rewrite the ICD manifest + apt graphics libs — done for pod 3).

**Third variant (pod3, 2026-07-23):** `/etc/vulkan/icd.d/nvidia_icd.json` present and correct, `graphics` capability present, libs installed — but sapien still dies with `vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent`. Cause: this image's Vulkan loader only reads `/usr/share/vulkan/icd.d/` (mesa ICDs only there), so enumeration falls back to llvmpipe, which lacks sapien's required device extensions. Fix: `cp /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/` — verify with `vulkaninfo --summary` (A6000 must appear) and an `env.reset` smoke test. Working pods have the manifest in BOTH paths; always mirror it.
