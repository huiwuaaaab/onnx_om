#!/usr/bin/env bash
# =============================================================================
# Qwen3-VL OM pipeline — pipe entry (multi-image stage pipeline)
#
# 6 workers, each OM preloaded once (OM_RESIDENT=1 by default).
#
# Usage:
#   RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch
#   MODE=full RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch 50
#
# Disable resident preload: OM_RESIDENT=0 RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch
# =============================================================================

set -euo pipefail

OM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${OM_DIR}/pipeline"
export OM_DIR PIPELINE_DIR

export OM_RESIDENT="${OM_RESIDENT:-1}"
export MSPROF_WRAP="${MSPROF_WRAP:-0}"

CPP_DAEMON="${PIPELINE_DIR}/om_resident_cpp/out/om_resident_daemon"
if [[ "${OM_RESIDENT}" == "1" && -x "${CPP_DAEMON}" ]]; then
  export OM_WORKER_SH="${OM_WORKER_SH:-${PIPELINE_DIR}/worker_cpp_resident.sh}"
elif [[ "${OM_RESIDENT}" == "1" ]]; then
  export OM_WORKER_SH="${OM_WORKER_SH:-${PIPELINE_DIR}/worker_resident.sh}"
else
  export OM_WORKER_SH="${OM_WORKER_SH:-${PIPELINE_DIR}/worker.sh}"
fi

# shellcheck source=pipeline/pipe.sh
source "${PIPELINE_DIR}/pipe.sh"
pipe_dispatch "$@"
