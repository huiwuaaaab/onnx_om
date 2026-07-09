#!/usr/bin/env bash
# Diagnose pyACL availability on MDC / AOS — run on board before pipe resident mode.
#
# Usage:
#   cd /opt/vlm/qwen3-vl
#   bash pipeline/diagnose_acl.sh

set -euo pipefail

OM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${OM_DIR}"

echo "=== pyACL diagnose (om=${OM_DIR}) ==="
echo

echo "[1] set_env.sh candidates"
for f in \
  /var/set_env.sh \
  /usr/local/Ascend/ascend-toolkit/set_env.sh \
  /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
  /usr/local/Ascend/nnrt/set_env.sh \
  /opt/ascend/nnrt/set_env.sh \
  "${HOME}/Ascend/ascend-toolkit/set_env.sh"
do
  [[ -f "${f}" ]] && echo "  OK  ${f}" || echo "  --  ${f}"
done
echo

echo "[2] source pipeline/acl_env.sh"
# shellcheck source=pipeline/acl_env.sh
source "${OM_DIR}/pipeline/acl_env.sh"
echo "  ACL_ENV_SOURCE=${ACL_ENV_SOURCE:-none}"
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}"
echo "  PYTHONPATH=${PYTHONPATH:-<empty>}"
echo

echo "[3] acl.so / acl*.so on system (first 20)"
if command -v find >/dev/null 2>&1; then
  find /usr/local/Ascend /var /opt/ascend /home/data -name 'acl*.so' 2>/dev/null | head -20 || true
else
  echo "  (find not available)"
fi
echo

echo "[4] python import acl test"
test_py() {
  local py="$1"
  [[ -x "${py}" || "${py}" == python3* ]] || return 0
  command -v "${py}" >/dev/null 2>&1 || return 0
  printf "  %-50s " "${py}"
  if out="$("${py}" -c "import acl; print(getattr(acl, '__file__', acl))" 2>&1)"; then
    echo "OK  ${out}"
    return 0
  else
    echo "FAIL"
    echo "       ${out}" | head -3
    return 1
  fi
}

ok=0
while IFS= read -r py; do
  [[ -n "${py}" ]] || continue
  if test_py "${py}"; then
    ok=1
    echo
    echo ">>> RECOMMENDED:"
    echo "    export ACL_PYTHON=${py}"
    echo "    export ASCEND_ENV_SH=${ACL_ENV_SOURCE:-/var/set_env.sh}"
    echo "    RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch"
    echo
  fi
done <<EOF
$(pick_acl_python 2>/dev/null || echo python3)
python3
python3.11
python3.10
python3.9
python3.8
python3.7
/usr/local/Ascend/ascend-toolkit/latest/bin/python3
/usr/local/Ascend/ascend-toolkit/latest/python/bin/python3
EOF

if [[ -d /usr/local/Ascend/ascend-toolkit/latest ]]; then
  while IFS= read -r py; do
    test_py "${py}" && ok=1
  done < <(find /usr/local/Ascend/ascend-toolkit/latest -maxdepth 6 -type f -name 'python3' 2>/dev/null | head -15)
fi

echo
if [[ "${ok}" -eq 0 ]]; then
  echo "=== RESULT: pyACL NOT available on this board ==="
  echo
  echo "msame (C++ ACL) works, but Python acl module is missing."
  echo "Pipe resident mode will fall back to msame-fallback:"
  echo "  - each vision job reloads OM (~2-5 min/job)"
  echo "  - 7 images x 50 steps = unusably slow"
  echo
  echo "Options:"
  echo "  A) Install CANN toolkit python / pyACL on MDC (ask platform admin)"
  echo "  B) Use serial mode for now (same per-step speed, simpler):"
  echo "       RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch"
  echo "  C) Copy acl.so + deps from a CANN toolkit host if versions match"
  echo
  echo "Check msame still works:"
  echo "  ./msame --help"
  exit 1
fi

echo "=== RESULT: pyACL available — set ACL_PYTHON above and rerun pipe ==="
exit 0
