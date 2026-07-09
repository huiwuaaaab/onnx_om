#!/usr/bin/env bash
# Single-process resident pool — one aclInit, all stage OMs (Gemma 11 workers).

set -euo pipefail

MANIFEST="${1:?manifest json path}"

OM_SCRIPT_DIR="${OM_SCRIPT_DIR:-${OM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
PY_POOL="${OM_SCRIPT_DIR}/pipeline/om_resident_pool.py"

export OM_SCRIPT_DIR

# shellcheck source=acl_env.sh
source "${OM_SCRIPT_DIR}/pipeline/acl_env.sh"

ACL_PY="$(pick_acl_python)"
echo "[$(date '+%H:%M:%S')][pool] starting resident pool python=${ACL_PY}" >&2
exec "${ACL_PY}" "${PY_POOL}" "${MANIFEST}"
