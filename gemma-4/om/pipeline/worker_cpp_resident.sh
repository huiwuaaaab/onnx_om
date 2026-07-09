#!/usr/bin/env bash
# C++ resident OM worker — AscendCL daemon (no pyACL required).

set -euo pipefail

WORKER_NAME="${1:?worker name}"
OM_PATH="${2:?om path}"
QUEUE_DIR="${3:?queue dir}"

OM_SCRIPT_DIR="${OM_SCRIPT_DIR:-${OM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DAEMON="${OM_RESIDENT_CPP_BIN:-${OM_SCRIPT_DIR}/pipeline/om_resident_cpp/out/om_resident_daemon}"

export OM_SCRIPT_DIR
export QWEN3_EXPORT_PROFILE="${QWEN3_EXPORT_PROFILE:-448_512}"
export EXPORT_PROFILE="${EXPORT_PROFILE:-${QWEN3_EXPORT_PROFILE}}"

# shellcheck source=acl_env.sh
source "${OM_SCRIPT_DIR}/pipeline/acl_env.sh"

if [[ ! -x "${DAEMON}" ]]; then
  echo "ERROR: C++ resident daemon not found: ${DAEMON}" >&2
  echo "  build on MDC: cd pipeline/om_resident_cpp && bash build.sh" >&2
  exit 1
fi

exec "${DAEMON}" "${WORKER_NAME}" "${OM_PATH}" "${QUEUE_DIR}"
