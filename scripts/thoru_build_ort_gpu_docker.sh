#!/bin/bash
# Build onnxruntime-gpu wheel inside CUDA devel Docker on ThorU (offline).
#
# Prereqs on ThorU:
#   /cus_app_data/guanxj/docker-images/cuda128-devel-u24-arm64.tar
#   /cus_app_data/guanxj/ort-build/onnxruntime/
#   /cus_app_data/guanxj/ort-build/wheels-arm64/
#   /cus_app_data/guanxj/ort-build/py312-include/
#
# Usage:
#   bash /cus_app_data/guanxj/scripts/thoru_build_ort_gpu_docker.sh
#   tail -f /cus_app_data/guanxj/ort-build/build.log
#
set -euo pipefail

DOCKER=${DOCKER:-/cus_app_data/guanxj/docker-bin/sbin/docker}
SUDO_PASS=${SUDO_PASS:-nvidia}
run_sudo() {
  echo "${SUDO_PASS}" | sudo -S "$@"
}
IMAGE=${IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04}
ORT_SRC=${ORT_SRC:-/cus_app_data/guanxj/ort-build/onnxruntime}
OUT_DIR=${OUT_DIR:-/cus_app_data/guanxj/ort-build/dist}
WHEELS=${WHEELS:-/cus_app_data/guanxj/qwen3-vl/wheels}
BUILD_WHEELS=${BUILD_WHEELS:-/cus_app_data/guanxj/ort-build/wheels-arm64}
CMAKE_MIRROR=${CMAKE_MIRROR:-/cus_app_data/guanxj/ort-build/cmake-mirror}
PY312_INC=${PY312_INC:-/cus_app_data/guanxj/ort-build/py312-include}
PY312_ARCH_INC=${PY312_ARCH_INC:-/cus_app_data/guanxj/ort-build/py312-arch-include}
HOST_SITE=${HOST_SITE:-/home/nvidia/.local/lib/python3.12/site-packages}
NUMPY_INC=${NUMPY_INC:-/home/nvidia/.local/lib/python3.12/site-packages/numpy/_core/include}
TAR=${TAR:-/cus_app_data/guanxj/docker-images/cuda128-devel-u24-arm64.tar}
JOBS=${JOBS:-2}
RESUME=${RESUME:-0}
CUDA_ARCH=${CUDA_ARCH:-"101-real;101-virtual"}
LOG=${LOG:-/cus_app_data/guanxj/ort-build/build.log}

mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

if [[ -f "$TAR" ]]; then
  SZ=$(stat -c%s "$TAR" 2>/dev/null || echo 0)
  if (( SZ < 5000000000 )); then
    echo "ERROR: $TAR looks incomplete (${SZ} bytes, expect ~5.1GB)"
    exit 1
  fi
fi

if ! run_sudo "$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1; then
  if [[ -f "$TAR" ]]; then
    echo "==> docker load $TAR"
    run_sudo "$DOCKER" load -i "$TAR"
  else
    echo "ERROR: image $IMAGE not found and no tar at $TAR"
    exit 1
  fi
fi

for p in "$ORT_SRC/build.sh" "$PY312_INC/Python.h" "$PY312_ARCH_INC/pyconfig.h" "$NUMPY_INC/numpy/arrayobject.h" "$CMAKE_MIRROR/github.com/abseil/abseil-cpp/archive/refs/tags/20250814.0.zip"; do
  [[ -e "$p" ]] || { echo "ERROR: missing $p"; exit 1; }
done

HOST_PY=$(readlink -f /usr/bin/python3.12)
HOST_PYLIB=$(readlink -f /usr/lib/aarch64-linux-gnu/libpython3.12.so.1.0)

echo "==> build onnxruntime-gpu in $IMAGE (jobs=$JOBS arch=$CUDA_ARCH)"
echo "==> log: $LOG"

run_sudo "$DOCKER" run --rm \
  --network host \
  -v "${ORT_SRC}:/src" \
  -v "${OUT_DIR}:/out" \
  -v "${WHEELS}:/wheels:ro" \
  -v "${BUILD_WHEELS}:/build-wheels:ro" \
  -v "${CMAKE_MIRROR}:/cmake-mirror:ro" \
  -v "${PY312_INC}:/usr/include/python3.12:ro" \
  -v "${PY312_ARCH_INC}:/usr/include/aarch64-linux-gnu/python3.12:ro" \
  -v "${HOST_PY}:/usr/bin/python3.12:ro" \
  -v /usr/lib/python3.12:/usr/lib/python3.12:ro \
  -v "${HOST_PYLIB}:${HOST_PYLIB}:ro" \
  -v /usr/lib/aarch64-linux-gnu/libpython3.12.so.1:/usr/lib/aarch64-linux-gnu/libpython3.12.so.1:ro \
  -v "${HOST_SITE}:/root/.local/lib/python3.12/site-packages:ro" \
  -v "${NUMPY_INC}:/numpy-include:ro" \
  -w /src \
  "$IMAGE" \
  bash -lc "
set -euo pipefail
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu
export PYTHONPATH=/root/.local/lib/python3.12/site-packages
export PYTHON_EXECUTABLE=/usr/bin/python3.12
ln -sf /usr/bin/python3.12 /usr/bin/python3
export CFLAGS=\"-I/usr/include/python3.12 \${CFLAGS:-}\"
export CXXFLAGS=\"-I/usr/include/python3.12 -Wno-error=deprecated-declarations \${CXXFLAGS:-}\"
export LDFLAGS=\"-L/usr/lib/aarch64-linux-gnu \${LDFLAGS:-}\"

python3.12 -m pip install -q --break-system-packages --no-index \
  --find-links=/build-wheels --find-links=/wheels \
  pip setuptools wheel cmake ninja packaging numpy

CCCL=\$(find /usr/local/cuda -path '*/include/cccl' -type d 2>/dev/null | head -1)
if [[ -z \"\$CCCL\" ]]; then
  CCCL=/usr/local/cuda/targets/sbsa-linux/include/cccl
fi
export CPLUS_INCLUDE_PATH=\"/usr/include/python3.12:/numpy-include:\${CCCL}:\${CPLUS_INCLUDE_PATH:-}\"

UPDATE_FLAG="--update"
if [[ "${RESUME}" == "1" ]]; then
  UPDATE_FLAG=""
fi

echo '==> toolchain'
python3.12 --version
cmake --version | head -1
ninja --version
nvcc --version | tail -1
g++ --version | head -1

./build.sh \
  --config Release \
  --allow_running_as_root \
  --skip_submodule_sync \
  --cmake_deps_mirror_dir /cmake-mirror \
  \${UPDATE_FLAG} --build \
  --parallel ${JOBS} \
  --skip_tests \
  --build_wheel \
  --enable_pybind \
  --cmake_generator Ninja \
  --use_cuda \
  --cuda_home /usr/local/cuda \
  --cudnn_home /usr/lib/aarch64-linux-gnu \
  --cmake_extra_defines Python_EXECUTABLE=/usr/bin/python3.12 \
  --cmake_extra_defines Python_INCLUDE_DIR=/usr/include/python3.12 \
  --cmake_extra_defines Python_LIBRARY=${HOST_PYLIB} \
  --cmake_extra_defines Python_NumPy_INCLUDE_DIR=/numpy-include \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES='${CUDA_ARCH}' \
  --cmake_extra_defines CMAKE_CXX_FLAGS=\"-I/usr/include/python3.12 -I\${CCCL} -Wno-error=deprecated-declarations\" \
  --cmake_extra_defines CMAKE_CUDA_FLAGS='--forward-unknown-to-host-compiler -Xcompiler=-Wno-strict-aliasing'

WHEEL=\$(ls -1 build/Linux/Release/dist/onnxruntime_gpu-*.whl | head -1)
cp -v \"\$WHEEL\" /out/
python3.12 -m pip install -q --break-system-packages --force-reinstall \"\$WHEEL\"
python3.12 -c \"import onnxruntime as ort; print('built', ort.__version__, ort.get_available_providers())\"
" 2>&1 | tee "$LOG"

echo "==> wheel in $OUT_DIR"
ls -lh "$OUT_DIR"/*.whl

echo "==> install on ThorU host:"
echo "python3 -m pip install --user --break-system-packages --force-reinstall $OUT_DIR/onnxruntime_gpu-*.whl"
