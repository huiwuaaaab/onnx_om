#!/usr/bin/env bash
# Build om_resident_daemon on MDC / CANN host (aarch64).
#
# Usage:
#   cd pipeline/om_resident_cpp
#   bash diagnose_cann_build.sh    # if build fails, run this first
#   bash build.sh
#
# If AOS has runtime-only CANN (no acl.h), build on a dev host with toolkit
# (same CANN version / aarch64) and scp out/om_resident_daemon to board.
#
#   scp out/om_resident_daemon user@<device-host>:/opt/vlm/qwen3-vl/pipeline/om_resident_cpp/out/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT}/build"

source_setenv() {
  local f
  for f in /var/set_env.sh \
           /usr/local/Ascend/ascend-toolkit/set_env.sh \
           /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
           /usr/local/Ascend/nnrt/set_env.sh \
           /usr/local/Ascend/nnrt/latest/set_env.sh \
           /opt/ascend/nnrt/set_env.sh; do
    if [[ -f "${f}" ]]; then
      # shellcheck source=/dev/null
      source "${f}"
      echo "sourced ${f}"
      return 0
    fi
  done
  return 1
}

source_setenv || true

find_acl_include_dir() {
  local root h parent
  # Explicit override
  if [[ -n "${ASCEND_INCLUDE:-}" && -f "${ASCEND_INCLUDE}/acl/acl.h" ]]; then
    echo "${ASCEND_INCLUDE}"
    return 0
  fi
  # Common roots after set_env
  for root in \
    "${DDK_PATH:-}" \
    /usr/local/Ascend/acllib \
    /usr/local/Ascend \
    /usr/local/Ascend/ascend-toolkit/latest \
    /usr/local/Ascend/ascend-toolkit \
    /usr/local/Ascend/nnrt/latest \
    /usr/local/Ascend/nnrt \
    /opt/ascend/nnrt/latest \
    /opt/ascend/nnrt
  do
    [[ -n "${root}" ]] || continue
    for h in \
      "${root}/include" \
      "${root}/acllib/include" \
      "${root}/runtime/include"
    do
      if [[ -f "${h}/acl/acl.h" ]]; then
        echo "${h}"
        return 0
      fi
    done
  done
  # Last resort: search filesystem
  h="$(find /usr/local/Ascend /opt/ascend /var -path '*/acl/acl.h' 2>/dev/null | head -1 || true)"
  if [[ -n "${h}" ]]; then
    parent="$(dirname "$(dirname "${h}")")"
    echo "${parent}"
    return 0
  fi
  return 1
}

find_acl_lib_dir() {
  local root
  if [[ -n "${ASCEND_LIB:-}" && -d "${ASCEND_LIB}" ]]; then
    echo "${ASCEND_LIB}"
    return 0
  fi
  if [[ -n "${NPU_HOST_LIB:-}" && -d "${NPU_HOST_LIB}" ]]; then
    echo "${NPU_HOST_LIB}"
    return 0
  fi
  for root in \
    "${DDK_PATH:-}" \
    /usr/local/Ascend/acllib \
    /usr/local/Ascend \
    /usr/local/Ascend/ascend-toolkit/latest \
    /usr/local/Ascend/nnrt/latest \
    /usr/local/Ascend/nnrt \
    /opt/ascend/nnrt/latest
  do
    [[ -n "${root}" ]] || continue
    for lib in \
      "${root}/lib64/stub" \
      "${root}/lib64" \
      "${root}/acllib/lib64/stub" \
      "${root}/acllib/lib64" \
      "${root}/runtime/lib64/stub"
    do
      if [[ -d "${lib}" ]]; then
        echo "${lib}"
        return 0
      fi
    done
  done
  return 1
}

ASCEND_INCLUDE="$(find_acl_include_dir || true)"
ASCEND_LIB="$(find_acl_lib_dir || true)"

if [[ -z "${ASCEND_INCLUDE}" || ! -f "${ASCEND_INCLUDE}/acl/acl.h" ]]; then
  echo "ERROR: acl/acl.h not found on this machine." >&2
  echo >&2
  echo "AOS 板端通常只有推理 runtime（msame 能跑），没有 CANN 开发头文件。" >&2
  echo "请在一台有 CANN toolkit 的机器上编译（aarch64，CANN 版本与板端一致），再 scp 二进制过来：" >&2
  echo >&2
  echo "  # 开发机（有 ascend-toolkit）" >&2
  echo "  cd pipeline/om_resident_cpp && bash build.sh" >&2
  echo "  scp out/om_resident_daemon user@<device-host>:/opt/vlm/qwen3-vl/pipeline/om_resident_cpp/out/" >&2
  echo >&2
  echo "或先运行: bash diagnose_cann_build.sh" >&2
  exit 1
fi

if [[ -z "${ASCEND_LIB}" ]]; then
  echo "ERROR: AscendCL stub lib dir not found. Set ASCEND_LIB=/path/to/lib64/stub" >&2
  exit 1
fi

echo "ASCEND_INCLUDE=${ASCEND_INCLUDE}"
echo "ASCEND_LIB=${ASCEND_LIB}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "ERROR: cmake not found (yum install cmake / apt install cmake)" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}" "${ROOT}/out"
cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
  -DASCEND_INCLUDE="${ASCEND_INCLUDE}" \
  -DASCEND_LIB="${ASCEND_LIB}"

cmake --build "${BUILD_DIR}" -j"$(nproc 2>/dev/null || echo 4)"

if [[ -x "${ROOT}/out/om_resident_daemon" ]]; then
  echo "OK: ${ROOT}/out/om_resident_daemon"
  file "${ROOT}/out/om_resident_daemon"
else
  echo "ERROR: build finished but out/om_resident_daemon missing" >&2
  exit 1
fi
