#!/bin/bash
# Full pi05_libero environment build from scratch. Consolidates the three stage
# scripts that built pilot3 (2026-09-10) into one; ~25 min, dominated by the
# 12GB checkpoint download (aria2, ~8 MB/s), which lives on /workspace so a stop does not wipe it.
set -x
export DEBIAN_FRONTEND=noninteractive PATH=/root/.local/bin:$PATH
apt-get update -qq
apt-get install -y -qq git git-lfs curl build-essential libegl1 libgles2 libglvnd0 libgl1 libosmesa6-dev libglew-dev patchelf
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /workspace
[ -d openpi ] || git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi && git checkout 15a9616 && git submodule update --init --recursive
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
deactivate
uv sync
cd /workspace && { [ -d interactive-vlas ] || git clone https://github.com/sttawm/interactive-vlas.git; }
cat > /workspace/interactive-vlas/pi05_libero/.openpi_env <<'ENV'
export LIBERO_VENV=/workspace/openpi/examples/libero/.venv
export LIBERO_PYTHONPATH=/workspace/openpi/third_party/libero
ENV
mkdir -p /workspace/.libero; rm -rf /root/.libero; ln -s /workspace/.libero /root/.libero
# openpi's checkpoint cache defaults to /root/.cache/openpi, which is container disk:
# a pod stop wipes it and the next serve_policy re-downloads 11.6 GiB serially at
# ~1 MB/s (3 h). Keep it on the volume, and prefetch it with aria2 (8 MB/s, ~25 min).
mkdir -p /workspace/.cache/openpi; rm -rf /root/.cache/openpi; mkdir -p /root/.cache; ln -s /workspace/.cache/openpi /root/.cache/openpi
apt-get install -y -qq aria2
CK=/workspace/.cache/openpi/openpi-assets/checkpoints/pi05_libero; mkdir -p $CK
curl -s "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_libero/&fields=items(name,size)&maxResults=200" \
 | python3 -c 'import sys,json
for i in json.load(sys.stdin)["items"]: print(i["size"], i["name"][len("checkpoints/pi05_libero/"):])' > /workspace/ckpt_manifest.txt
: > /workspace/aria2_in.txt
while read sz rel; do printf '%s\n  dir=%s\n  out=%s\n' "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/$rel" "$CK/$(dirname $rel)" "$(basename $rel)" >> /workspace/aria2_in.txt; done < /workspace/ckpt_manifest.txt
aria2c -i /workspace/aria2_in.txt -j 8 -x 8 -s 8 -k 8M -c --file-allocation=none --max-tries=20 --summary-interval=60 --console-log-level=warn
while read sz rel; do [ "$(stat -c %s "$CK/$rel")" = "$sz" ] || { echo "CHECKPOINT SIZE MISMATCH $rel"; exit 3; }; done < /workspace/ckpt_manifest.txt
. /workspace/interactive-vlas/pi05_libero/.openpi_env
echo N | MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$LIBERO_PYTHONPATH" "$LIBERO_VENV/bin/python" -c "import libero.libero" >/dev/null 2>&1
# HARD GATE: real render
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$LIBERO_PYTHONPATH" "$LIBERO_VENV/bin/python" -c "
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np,os
bd=benchmark.get_benchmark_dict()['libero_90']()
t=bd.get_task(31); b=os.path.join(get_libero_path('bddl_files'),t.problem_folder,t.bddl_file)
e=OffScreenRenderEnv(bddl_file_name=b,camera_heights=256,camera_widths=256); e.seed(7); e.reset()
im=e.step(np.zeros(7))[0]['agentview_image']; print('RENDER_MEAN',round(float(im.mean()),1)); e.close()"
cat > /workspace/start_server.sh <<'SRV'
#!/bin/bash
cd /workspace/openpi
exec .venv/bin/python3 scripts/serve_policy.py --env LIBERO --port 8000 >> /workspace/server.log 2>&1
SRV
chmod +x /workspace/start_server.sh
setsid nohup /workspace/start_server.sh >/dev/null 2>&1 < /dev/null &
echo "BUILD DONE (server starting, downloads checkpoint on first run)"
