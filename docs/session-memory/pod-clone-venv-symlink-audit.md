---
name: pod-clone-venv-symlink-audit
description: "before tarring/cloning a pod workspace, audit for venv symlinks into /root — tar archives dangling links; restore to the SAME path on the target"
metadata: 
  node_type: memory
  type: project
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
---

2026-07-22: cloning pod1→pod2 via workspace tarball silently shipped a dangling
`.venv-gen -> /root/venv-gen` symlink (the container-disk venv pattern recurred
even after the Jul-18 "real dirs" rebuild — only .venv-score was actually made
real). The search worker crash-looped on a missing interpreter.

**Why:** tar archives the link, not the target; /root is per-container and never
in workspace tarballs.

**How to apply:** before any pod clone, run `find /workspace -maxdepth 3 -type l
-exec ls -la {} +` and either tar with `-h` (follow links) or relay the /root
target separately and untar it to the IDENTICAL path on the destination (venvs
are path-anchored — same-path restore works instantly through the existing
symlink). Long-term fix: rebuild venvs as real dirs under /workspace and delete
the /root copies. Related: [[runpod-podstate-backup]], [[gitignored-data-payload]].
