#!/usr/bin/env bash
# Find CANN acllib include/lib paths for building om_resident_daemon.
#
# Usage: bash diagnose_cann_build.sh

set -eu

echo "=== CANN build diagnose (AOS / MDC) ==="
echo

echo "[set_env.sh]"
for f in /var/set_env.sh \
         /usr/local/Ascend/ascend-toolkit/set_env.sh \
         /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
         /usr/local/Ascend/acllib/set_env.sh \
         /usr/local/Ascend/nnrt/set_env.sh \
         /usr/local/Ascend/nnrt/latest/set_env.sh \
         /opt/ascend/nnrt/set_env.sh; do
  if [[ -f "${f}" ]]; then
    echo "  OK  ${f}"
  else
    echo "  --  ${f}"
  fi
done
echo

ENV_LOADED=0
for f in /var/set_env.sh \
         /usr/local/Ascend/ascend-toolkit/set_env.sh \
         /usr/local/Ascend/nnrt/set_env.sh \
         /opt/ascend/nnrt/set_env.sh; do
  if [[ -f "${f}" ]]; then
    # shellcheck source=/dev/null
    source "${f}" && ENV_LOADED=1 && echo "sourced ${f}" && break
  fi
done
[[ "${ENV_LOADED}" -eq 1 ]] || echo "WARN: no set_env.sh sourced"
echo

echo "DDK_PATH=${DDK_PATH-<unset>}"
echo "NPU_HOST_LIB=${NPU_HOST_LIB-<unset>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH-<unset>}"
echo

echo "[Ascend acllib on board (from msame ldd)]"
for p in \
  /usr/local/Ascend/acllib/include/acl/acl.h \
  /usr/local/Ascend/acllib/lib64/libascendcl.so \
  /usr/local/Ascend/acllib/lib64/stub/libascendcl.so; do
  if [[ -e "${p}" ]]; then
    echo "  OK  ${p}"
  else
    echo "  --  ${p}"
  fi
done
echo

echo "[acl.h search]"
ACL_FOUND=0
ACL_TMP="$(mktemp /tmp/acl_h.XXXXXX 2>/dev/null || echo /tmp/acl_h.$$)"
find /usr/local/Ascend /opt/ascend /var /home/data -name 'acl.h' 2>/dev/null | head -10 > "${ACL_TMP}" || true
while IFS= read -r h; do
  [[ -n "${h}" ]] || continue
  echo "  ${h}"
  ACL_FOUND=1
done < "${ACL_TMP}"
rm -f "${ACL_TMP}"

if [[ "${ACL_FOUND}" -eq 0 ]]; then
  echo "  NOT FOUND"
  echo
  echo ">>> AOS 板端通常只有推理 runtime（msame 能跑），没有 CANN 开发头文件。"
  echo ">>> 请在有 CANN toolkit 的开发机编译 om_resident_daemon，再 scp 到板子："
  echo ">>>   scp out/om_resident_daemon root@AOS:.../pipeline/om_resident_cpp/out/"
fi
echo

echo "[libascendcl.so search]"
LIB_TMP="$(mktemp /tmp/acl_lib.XXXXXX 2>/dev/null || echo /tmp/acl_lib.$$)"
find /usr/local/Ascend /opt/ascend /var /home/data -name 'libascendcl.so' 2>/dev/null | head -10 > "${LIB_TMP}" || true
while IFS= read -r lib; do
  [[ -n "${lib}" ]] || continue
  echo "  ${lib}"
done < "${LIB_TMP}"
rm -f "${LIB_TMP}"
echo

echo "[msame on board]"
if command -v msame >/dev/null 2>&1; then
  echo "  PATH: $(command -v msame)"
  file "$(command -v msame)" || true
fi
for m in /home/mdc/guanxj/qwen3-vl/msame \
         /home/mdc/guanxj/qwen3-vl/om/msame \
         ../msame ../../msame ./msame; do
  if [[ -x "${m}" ]]; then
    file "${m}" || true
    echo "  ${m}"
  fi
done
echo

echo "[cmake / g++]"
command -v cmake >/dev/null 2>&1 && echo "  cmake: $(command -v cmake)" || echo "  cmake: NOT FOUND"
command -v g++ >/dev/null 2>&1 && echo "  g++:   $(command -v g++)" || echo "  g++:   NOT FOUND"
