#!/usr/bin/env bash
# =============================================================================
# Gemma-4 OM pipeline — pipe entry (main chain, no assistant)
#
# 11 worker processes (persistent), OM load/unload per job (ctypes).
# Requires MODE=main_decode.
#
# Usage:
#   MODE=main_decode RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch
#   MODE=main_decode RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch 50
#
# Disable ctypes: OM_RESIDENT=0 MODE=main_decode RUN_MSAME=1 bash run_om_pipeline_pipe.sh ./batch
# Keep OM loaded in worker (may OOM on 11 stages): OM_PER_JOB_LOAD=0 ...
# =============================================================================

set -euo pipefail

OM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${OM_DIR}/pipeline"
export OM_DIR PIPELINE_DIR

export OM_RESIDENT="${OM_RESIDENT:-1}"
export OM_PER_JOB_LOAD="${OM_PER_JOB_LOAD:-1}"
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
