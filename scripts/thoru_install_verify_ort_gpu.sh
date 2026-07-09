#!/bin/bash
# Install self-built onnxruntime-gpu wheel on ThorU and verify CUDA EP.
set -euo pipefail

WHEEL=${WHEEL:-/cus_app_data/guanxj/ort-build/dist/onnxruntime_gpu-1.27.0-cp312-cp312-linux_aarch64.whl}
QWEN=${QWEN:-/cus_app_data/guanxj/qwen3-vl}
SITE=${SITE:-/cus_app_data/guanxj/py312-site-packages}
WHEELS=${WHEELS:-/cus_app_data/guanxj/qwen3-vl/wheels}

if [[ ! -f "$WHEEL" ]]; then
  WHEEL=$(ls -1 /cus_app_data/guanxj/ort-build/dist/onnxruntime_gpu-*.whl | head -1)
fi
[[ -f "$WHEEL" ]] || { echo "ERROR: wheel not found"; exit 1; }

echo "==> uninstall CPU onnxruntime (if any)"
python3 -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
rm -rf "$SITE"
mkdir -p "$SITE"

echo "==> install $WHEEL -> $SITE"
python3 -m pip install --break-system-packages --force-reinstall \
  --no-index --find-links="$WHEELS" --target="$SITE" "$WHEEL"

export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/thor/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "==> import check"
PYTHONPATH="$SITE" LD_LIBRARY_PATH="/usr/local/cuda-12.8/thor/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu" python3 - <<'PY'
import onnxruntime as ort
print("version:", ort.__version__)
print("providers:", ort.get_available_providers())
PY

echo "==> CUDA session on vision_448.onnx"
PYTHONPATH="$SITE" LD_LIBRARY_PATH="/usr/local/cuda-12.8/thor/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu" python3 - <<PY
import json, time, numpy as np, onnxruntime as ort
from pathlib import Path

qwen = Path("$QWEN")
model = qwen / "onnx_export/vision_448.onnx"
meta = json.loads((qwen / "om/vision_bin/meta.json").read_text())
spec = meta["tensors"]["pixel_values"]
pv = np.fromfile(qwen / "om/vision_bin/pixel_values.bin", dtype=np.float16).reshape(tuple(spec["shape"]))

CUDA_OPTS = {"device_id": "0", "cudnn_conv_algo_search": "DEFAULT"}
opts = ort.SessionOptions(); opts.log_severity_level = 2
for label, prov in [
    ("CUDA+DEFAULT", [("CUDAExecutionProvider", CUDA_OPTS), "CPUExecutionProvider"]),
    ("CPU", ["CPUExecutionProvider"]),
]:
    print("\\n===", label, "===")
    try:
        sess = ort.InferenceSession(str(model), opts, providers=prov)
        print("active:", sess.get_providers())
        t0 = time.time()
        out = sess.run(None, {"hidden_states": np.ascontiguousarray(pv, np.float16)})
        print("run OK outputs:", len(out), "time:", round(time.time()-t0, 3), "s")
    except Exception as e:
        print("FAILED:", type(e).__name__, e)
PY
