#!/usr/bin/env bash
# Resident OM worker — model preloaded once via om_resident_worker.py (pyACL).
# Same FIFO job protocol as worker.sh.

set -euo pipefail

WORKER_NAME="${1:?worker name}"
OM_PATH="${2:?om path}"
QUEUE_DIR="${3:?queue dir}"

OM_SCRIPT_DIR="${OM_SCRIPT_DIR:-${OM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
PY_WORKER="${OM_SCRIPT_DIR}/pipeline/om_resident_worker.py"

export MSAME_BIN="${MSAME_BIN:-${OM_SCRIPT_DIR}/msame}"
export MSPROF_BIN="${MSPROF_BIN:-/var/msprof}"
export MSPROF_WRAP="${MSPROF_WRAP:-0}"
export OM_SCRIPT_DIR
export QWEN3_EXPORT_PROFILE="${QWEN3_EXPORT_PROFILE:-448_512}"
export EXPORT_PROFILE="${EXPORT_PROFILE:-${QWEN3_EXPORT_PROFILE}}"

# Same CANN env as msame — required for `import acl`.
# shellcheck source=acl_env.sh
source "${OM_SCRIPT_DIR}/pipeline/acl_env.sh"

ACL_PY="$(pick_acl_python)"
if acl_import_ok; then
  echo "[$(date '+%H:%M:%S')][resident:${WORKER_NAME}] pyACL ok python=${ACL_PY} env=${ACL_ENV_SOURCE:-unknown}" >&2
else
  echo "[$(date '+%H:%M:%S')][resident:${WORKER_NAME}] pyACL unavailable; worker will try ctypes libascendcl.so" >&2
  if [[ "${OM_RESIDENT_REQUIRE_ACL:-0}" == "1" ]]; then
    echo "ERROR: OM_RESIDENT_REQUIRE_ACL=1 but pyACL unavailable" >&2
    exit 1
  fi
fi

exec "${ACL_PY}" "${PY_WORKER}" "${WORKER_NAME}" "${OM_PATH}" "${QUEUE_DIR}"
